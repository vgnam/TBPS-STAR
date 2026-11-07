"""RA-TBPS on a frozen CLIP TBPS model in the Pretrain-then-Adapt setting.

The model (e.g. MLLM4Text-ReID, pretrained on synthetic captions only) never
sees a labelled target pair. The memory is the target test split itself:
every caption j is stored with its pseudo image p_j (its text-to-image top-1),
flagged reliable when the agreement is mutual (j is also in the image-to-text
top-r of p_j, as in UATTA). A query never retrieves its own entry.

Scores for a query q and gallery image g, on top of the frozen cosine s(q, g):
  query side      RA-VLA evidence: the K memory captions nearest to q vote with
                  their pseudo images, + beta * sum_k w_k cos(i_k, g)
  candidate side  captions grounded on g vote for it,
                  + gamma * mean_{j: p_j = g, j != q} cos(q, t_j)
  learned         a grounded scorer g_theta trained on the reliable pseudo pairs
                  (listwise loss + contextual adherence margin), fed only
                  evidence relations, never s(q, g) itself, so it cannot learn
                  to copy the frozen ranking that produced its labels.
                  Final logits: s / tau + g_theta.

usage: python -m ratbps.clip_tta --npz results/kaggle_uatta_v1/clip_zeroshot_rstp_test.npz
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import GroundingBlock, adherence_loss, ranking_loss

dev = "cuda" if torch.cuda.is_available() else "cpu"
N_REL = 6


def contextual_sensitivity(s_ret, s_rand):
    """RA-VLA eq. (8) on each query's candidate logits, centred per query."""
    a = s_ret - s_ret.mean(1, keepdim=True)
    b = s_rand - s_rand.mean(1, keepdim=True)
    return ((a - b).norm(dim=1) / a.norm(dim=1).clamp(min=1e-6)).mean().item()


def load(path):
    z = np.load(path)
    t = F.normalize(torch.as_tensor(z["txt_feat"], device=dev).float(), dim=-1)
    i = F.normalize(torch.as_tensor(z["img_feat"], device=dev).float(), dim=-1)
    return dict(txt=t, img=i, qid=torch.as_tensor(z["qid"], device=dev).long(),
                gid=torch.as_tensor(z["gid"], device=dev).long())


@torch.no_grad()
def metrics(c, score_fn, chunk=512):
    """IRRA's t2i protocol: R1/R5/R10/mAP/mINP over the full gallery.
    score_fn(q_idx) -> (B, G) scores."""
    cmc, ap, inp = [], [], []
    for s in range(0, len(c["txt"]), chunk):
        q = torch.arange(s, min(len(c["txt"]), s + chunk), device=dev)
        match = c["gid"][score_fn(q).argsort(1, descending=True)] == c["qid"][q, None]
        hits = match.cumsum(1)
        cmc.append(hits[:, :10] > 0)
        rank = torch.arange(1, match.shape[1] + 1, device=dev)
        ap.append(((hits / rank) * match).sum(1) / match.sum(1).clamp(min=1))
        last = match.shape[1] - 1 - match.flip(1).float().argmax(1)
        inp.append(hits.gather(1, last[:, None])[:, 0] / (last + 1.0))
    cmc = torch.cat(cmc).float().mean(0) * 100
    return dict(R1=cmc[0].item(), R5=cmc[4].item(), R10=cmc[9].item(),
                mAP=torch.cat(ap).mean().item() * 100, mINP=torch.cat(inp).mean().item() * 100)


@torch.no_grad()
def build_memory(c, r):
    S = c["txt"] @ c["img"].T
    top2 = S.topk(2, dim=1)
    p = top2.indices[:, 0]
    i2t_rank = (S[:, p].T > S[torch.arange(len(p), device=dev), p][:, None]).sum(1)  # how many captions beat j on p_j
    mutual = i2t_rank < r
    conf = top2.values[:, 0] - top2.values[:, 1]
    prec = lambda m: (c["gid"][p[m]] == c["qid"][m]).float().mean().item() * 100
    print(f"memory: {len(p)} captions, {mutual.sum().item()} mutual (top-{r}); "
          f"pseudo precision all {prec(torch.ones_like(mutual)):.1f}%, mutual {prec(mutual):.1f}%", flush=True)
    return dict(S=S, p=p, mutual=mutual, conf=conf)


@torch.no_grad()
def query_evidence(c, mem, k, only_mutual=True, rand=False, seed=0):
    """K memory captions per query (never itself), most similar first."""
    Q = len(c["txt"])
    sim = c["txt"] @ c["txt"].T
    sim.fill_diagonal_(-2)
    if only_mutual:
        sim[:, ~mem["mutual"]] = -2
    if rand:
        g = torch.Generator(device=dev).manual_seed(seed)
        pool = torch.nonzero(mem["mutual"] if only_mutual else torch.ones(Q, dtype=torch.bool, device=dev))[:, 0]
        idx = pool[torch.randint(len(pool), (Q, k), device=dev, generator=g)]
        own = idx == torch.arange(Q, device=dev)[:, None]
        idx = torch.where(own, pool[(torch.searchsorted(pool, idx) + 1) % len(pool)], idx)
        v = sim.gather(1, idx)
        v, o = v.sort(1, descending=True)
        return idx.gather(1, o), v
    v, idx = sim.topk(k, dim=1)
    return idx, v


@torch.no_grad()
def candidate_side(c, mem, only_mutual=True):
    """mean_{j: p_j = g, j != q} cos(q, t_j) and the count, both (Q, G)."""
    Q, G = len(c["txt"]), len(c["img"])
    use = mem["mutual"] if only_mutual else torch.ones(Q, dtype=torch.bool, device=dev)
    A = torch.zeros(Q, G, device=dev)
    A[use.nonzero()[:, 0], mem["p"][use]] = 1.0
    tt = c["txt"] @ c["txt"].T
    tot = tt @ A
    cnt = A.sum(0, keepdim=True).expand(Q, G).clone()
    self_q = use.nonzero()[:, 0]
    tot[self_q, mem["p"][self_q]] -= 1.0  # cos(q, q)
    cnt[self_q, mem["p"][self_q]] -= 1.0
    has = cnt > 0
    mean = tot / cnt.clamp(min=1)
    # images no caption grounds get the query's average vote: no evidence, no bonus or penalty
    neutral = (mean * has).sum(1, keepdim=True) / has.sum(1, keepdim=True).clamp(min=1)
    return torch.where(has, mean, neutral.expand_as(mean)), cnt


def zscore(x):
    return (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True).clamp(min=1e-6)


def free_score(c, mem, ev, cs, beta, gamma, tau_w=0.05):
    """z(s) + beta z(query-side votes) + gamma z(candidate-side votes); each term
    standardised over the gallery per query so beta and gamma are relative weights."""
    idx, v = ev
    w = F.softmax(v / tau_w, dim=1)
    ev_img = c["img"][mem["p"][idx]]  # (Q, K, d)

    def fn(q):
        s = zscore(c["txt"][q] @ c["img"].T)
        if beta:
            s = s + beta * zscore(torch.einsum("bk,bkd,gd->bg", w[q], ev_img[q], c["img"]))
        if gamma:
            s = s + gamma * zscore(cs[0][q])
        return s
    return fn


class ClipScorer(nn.Module):
    """Grounded scorer over evidence relations only (no s(q, g) input)."""

    def __init__(self, k, d=128, heads=4):
        super().__init__()
        self.k = k
        self.pair = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.rel = nn.Sequential(nn.Linear(N_REL, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([GroundingBlock(d, heads) for _ in range(k)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, pair, rel):
        h = self.pair(pair).unsqueeze(1)
        e = self.rel(rel)
        for l, block in enumerate(self.blocks):
            h = block(h, e[:, self.k - 1 - l].unsqueeze(1))  # most relevant in the last block
        return self.out(h[:, 0]).squeeze(-1)


def scorer_inputs(c, mem, ev, cs, q, cand):
    """pair (B*N, 2) and rel (B*N, K, 6) for queries q and their candidates."""
    idx, v = ev[0][q], ev[1][q]
    B, N = cand.shape
    K = idx.shape[1]
    tq = c["txt"][q]
    gi = c["img"][cand]  # (B, N, d)
    et, ei = c["txt"][idx], c["img"][mem["p"][idx]]  # (B, K, d)
    rel = torch.stack([
        v[:, None].expand(B, N, K),
        torch.einsum("bnd,bkd->bnk", gi, ei),
        (tq[:, None] * ei).sum(-1)[:, None].expand(B, N, K),
        torch.einsum("bnd,bkd->bnk", gi, et),
        mem["conf"][idx][:, None].expand(B, N, K),
        (mem["p"][idx][:, None] == cand[:, :, None]).float(),
    ], -1).reshape(B * N, K, N_REL)
    pair = torch.stack([cs[0][q[:, None], cand], torch.log1p(cs[1][q[:, None], cand])], -1).reshape(B * N, 2)
    return pair, rel


def train_scorer(c, mem, ev, ev_rand_fn, cs, cand, k, steps, lam, margin, bs=128, lr=3e-4, seed=0):
    torch.manual_seed(seed)
    g = ClipScorer(k).to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=lr, weight_decay=1e-4)
    train_q = mem["mutual"].nonzero()[:, 0]
    pos_all = cand == mem["p"][:, None]
    for it in range(steps):
        q = train_q[torch.randint(len(train_q), (bs,), device=dev)]
        cq = cand[q]
        # g alone must recover the pseudo label from evidence; s(q, g) joins only at test time
        pair, rel = scorer_inputs(c, mem, ev, cs, q, cq)
        l_rel = ranking_loss(g(pair, rel).view(cq.shape), pos_all[q])
        loss = l_rel
        if lam:
            ev_r = ev_rand_fn(it)
            pair_r, rel_r = scorer_inputs(c, mem, ev_r, cs, q, cq)
            l_irr = ranking_loss(g(pair_r, rel_r).view(cq.shape), pos_all[q])
            loss = loss + lam * adherence_loss(l_rel, l_irr, margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 500 == 0:
            print(f"  scorer step {it + 1}: loss {loss.item():.4f} rank {l_rel.item():.4f}", flush=True)
    return g.eval()


@torch.no_grad()
def scorer_logits(g, c, mem, ev, cs, cand, tau, chunk=256):
    out = []
    for s in range(0, len(cand), chunk):
        q = torch.arange(s, min(len(cand), s + chunk), device=dev)
        cq = cand[q]
        base = (c["txt"][q][:, None] * c["img"][cq]).sum(-1) / tau
        pair, rel = scorer_inputs(c, mem, ev, cs, q, cq)
        out.append(base + g(pair, rel).view(cq.shape))
    return torch.cat(out)


def rerank_fn(c, cand, logits):
    def fn(q):
        s = c["txt"][q] @ c["img"].T
        top = logits[q]
        return s.scatter(1, cand[q], top - top.min(1, keepdim=True)[0] + 10.0)  # cosine < 10
    return fn


def timed(fn):
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    out = fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--r", type=int, default=5, help="mutual top-r, UATTA's k_sample_selection")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--sweep", default="0,0.1,0.25,0.5,1")
    ap.add_argument("--topn", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    c = load(args.npz)
    print(f"{args.npz}: {len(c['txt'])} queries, {len(c['img'])} gallery images", flush=True)
    res = dict(rows={}, sweep={}, k_table={}, diag={})
    show = lambda name, m: print(f"{name:<34} " + "  ".join(f"{k} {v:6.2f}" for k, v in m.items()), flush=True)

    def row(name, fn):
        res["rows"][name] = m = metrics(c, fn)
        show(name, m)

    zs = lambda q: c["txt"][q] @ c["img"].T
    row("zero-shot", zs)
    mem = build_memory(c, args.r)
    res["diag"]["mutual"] = mem["mutual"].sum().item()
    res["diag"]["pseudo_prec_mutual"] = (c["gid"][mem["p"][mem["mutual"]]] == c["qid"][mem["mutual"]]).float().mean().item() * 100

    ev = query_evidence(c, mem, max(int(x) for x in args.ks.split(",")))
    cut = lambda e, k: (e[0][:, :k], e[1][:, :k])
    cs = candidate_side(c, mem)
    evk = cut(ev, args.k)

    # training-free rows at the a-priori defaults
    row(f"query-side evidence (b={args.beta})", free_score(c, mem, evk, cs, args.beta, 0))
    row(f"candidate-side evidence (g={args.gamma})", free_score(c, mem, evk, cs, 0, args.gamma))
    row("RA-TBPS-free (both)", free_score(c, mem, evk, cs, args.beta, args.gamma))
    ev_rand = query_evidence(c, mem, args.k, rand=True, seed=args.seed)
    row("  random query evidence", free_score(c, mem, ev_rand, cs, args.beta, args.gamma))
    ev_all = cut(query_evidence(c, mem, args.k, only_mutual=False), args.k)
    row("  memory without mutual filter", free_score(c, mem, ev_all, candidate_side(c, mem, only_mutual=False),
                                                     args.beta, args.gamma))

    # oracle sweep (uses test labels, reported separately)
    grid = [float(x) for x in args.sweep.split(",")]
    for b in grid:
        for gm in grid:
            m = metrics(c, free_score(c, mem, evk, cs, b, gm))
            res["sweep"][f"b{b}_g{gm}"] = m
    best = max(res["sweep"].items(), key=lambda kv: kv[1]["R1"])
    print(f"oracle sweep best {best[0]}: R1 {best[1]['R1']:.2f} mAP {best[1]['mAP']:.2f}", flush=True)

    # learned scorer
    cand = (c["txt"] @ c["img"].T).topk(args.topn, dim=1).indices

    def rand_source(k, n=50):
        """Random-evidence draws for the adherence term, built once per K."""
        cache = {}

        def draw(it):
            if it % n not in cache:
                cache[it % n] = query_evidence(c, mem, k, rand=True, seed=args.seed + 1 + it % n)
            return cache[it % n]
        return draw

    scorers = {}
    for name, lam in (("RA-TBPS (learned)", args.lam), ("  no adherence", 0.0)):
        print(f"training {name.strip()} (K={args.k}, lambda={lam})", flush=True)
        scorers[name] = train_scorer(c, mem, evk, rand_source(args.k), cs, cand, args.k, args.steps, lam,
                                     args.margin, seed=args.seed)
        lg = scorer_logits(scorers[name], c, mem, evk, cs, cand, args.tau)
        row(name, rerank_fn(c, cand, lg))
        lg_r = scorer_logits(scorers[name], c, mem, ev_rand, cs, cand, args.tau)
        row(name + " / random evidence", rerank_fn(c, cand, lg_r))
        res["diag"][f"S_ctx {name.strip()}"] = contextual_sensitivity(lg, lg_r)
        print(f"  S_ctx {res['diag'][f'S_ctx {name.strip()}']:.3f}", flush=True)

    # K table: training-free at defaults and learned, with time per query
    for k in (int(x) for x in args.ks.split(",")):
        e = cut(ev, k)
        fn = free_score(c, mem, e, cs, args.beta, args.gamma)
        m, t = timed(lambda: metrics(c, fn))
        _, t0 = timed(lambda: metrics(c, zs))
        g = scorers["RA-TBPS (learned)"] if k == args.k else train_scorer(
            c, mem, e, rand_source(k), cs, cand, k, args.steps, args.lam, args.margin, seed=args.seed)
        lg, tl = timed(lambda: scorer_logits(g, c, mem, e, cs, cand, args.tau))
        ml = metrics(c, rerank_fn(c, cand, lg))
        Q = len(c["txt"])
        res["k_table"][k] = dict(free=m, learned=ml, ms_free=(t - t0) / Q * 1e3, ms_learned=tl / Q * 1e3)
        print(f"K={k}: free R1 {m['R1']:.2f} mAP {m['mAP']:.2f} | learned R1 {ml['R1']:.2f} mAP {ml['mAP']:.2f} "
              f"| {res['k_table'][k]['ms_learned']:.3f} ms/query", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print("->", args.out)


if __name__ == "__main__":
    main()
