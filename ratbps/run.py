"""Train RA-TBPS on the source cache, then adapt to each target split.

Training simulates test-time adaptation inside the source domain, mirroring
RA-VLA's held-out-task protocol: source identities are split in two halves.
Half A stays annotated; half B plays the unlabelled target, so its memory
entries are pseudo-pairs built exactly as at test time. R and g train on
queries from B against memory = annotated(A) + pseudo(B).

At test time nothing is updated: memory = annotated(all source) +
pseudo(target), and the target's labels are read only by the metrics (and to
log pseudo-pair precision).

usage: python -m ratbps.run --src emb/aptm_cuhk_train.npz \
           --target cuhk=emb/aptm_cuhk_test.npz --target rstp=emb/aptm_rstp_test.npz \
           --out results/ratbps
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import memory as M
from .model import Retriever, Scorer, adherence_loss, alignment_loss, ranking_loss, relation_features

dev = "cuda" if torch.cuda.is_available() else "cpu"
SRC, TGT = 0, 1  # memory domains; queries being ranked always belong to TGT


def load(path):
    d = dict(np.load(path))
    print(f"{path}: {len(d['qid'])} captions, {len(d['gid'])} images, top-{d['cand'].shape[1]} candidates",
          flush=True)
    return d


def gpu(d, queries=None, n=None):
    """Ranking view of a cache on the device, optionally restricted to some
    queries (re-indexed 0..len-1) and their first n candidates."""
    q = slice(None) if queries is None else queries
    c = slice(None) if n is None else slice(0, n)
    t = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)  # noqa: E731
    return dict(cand=t(d["cand"][q, c]).long(), itm=t(d["cand_itm"][q, c]), itc=t(d["cand_itc"][q, c]),
                cross=t(d["cand_cross"][q, c]), txt=t(d["txt_feat"][q]), img=t(d["img_feat"]),
                gid=t(d["gid"]).long(), qid=t(d["qid"][q]).long())


def to_dev(mem):
    t = {k: torch.as_tensor(v, device=dev) for k, v in mem.items()}
    t["cross"] = t["cross"].half()
    t["pid"] = t["pid"].long()
    return t


def pairs(c, q):
    """Flattened (query, candidate) pairs for query positions q: P = len(q) * N."""
    B, N = len(q), c["cand"].shape[1]
    cand = c["cand"][q]
    return dict(cross=c["cross"][q].reshape(B * N, -1),
                txt=c["txt"][q][:, None].expand(B, N, -1).reshape(B * N, -1),
                img=c["img"][cand].reshape(B * N, -1),
                base=torch.stack([c["itm"][q], c["itc"][q]], -1).reshape(B * N, 2))


@torch.no_grad()
def keys(R, c, chunk=512):
    """Retriever keys of every (query, candidate) pair: (Q, N, dz)."""
    Q, N = c["cand"].shape
    out = []
    for s in range(0, Q, chunk):
        p = pairs(c, torch.arange(s, min(Q, s + chunk), device=dev))
        out.append(R(p["cross"], p["txt"], p["img"]).view(-1, N, R.head.out_features))
    return torch.cat(out)


@torch.no_grad()
def encode_memory(R, mem, chunk=16384):
    return torch.cat([R(mem["cross"][s:s + chunk], mem["txt"][s:s + chunk], mem["img"][s:s + chunk])
                      for s in range(0, len(mem["cross"]), chunk)])


@torch.no_grad()
def retrieve(zq, owner_q, zm, owner_m, k, chunk=4096):
    """Top-k memory entries per key, never a query's own entry. Returns (idx, sim)."""
    idx, sim = [], []
    for s in range(0, len(zq), chunk):
        v, i = (zq[s:s + chunk] @ zm.T).topk(k + 1, dim=1)
        v = v.masked_fill(owner_m[i] == owner_q[s:s + chunk, None], -2.0)
        v, j = v.topk(k, dim=1)
        idx.append(i.gather(1, j))
        sim.append(v)
    return torch.cat(idx), torch.cat(sim)


def evidence_for(R, c, mem, zm, k, keep=None):
    """Retrieved evidence for every pair of cache c: idx, sim (Q, N, k)."""
    Q, N = c["cand"].shape
    zq = keys(R, c).reshape(Q * N, -1)
    own = torch.as_tensor(M.owner_ids(TGT, np.arange(Q)), device=dev).repeat_interleave(N)
    mz, mo = (zm, mem["owner"]) if keep is None else (zm[keep], mem["owner"][keep])
    i, v = retrieve(zq, own, mz, mo, k)
    if keep is not None:
        i = torch.nonzero(keep)[:, 0][i]
    return i.view(Q, N, k), v.view(Q, N, k)


def evidence(mem, idx, sim, txt, img):
    """Evidence tokens for pairs. idx/sim (P,K); txt/img (P,256)."""
    rel = relation_features(txt, img, mem["txt"][idx], mem["img"][idx], sim, mem["pseudo"][idx])
    return mem["cross"][idx], rel


# ---------------------------------------------------------------- training

def train_retriever(R, c, mem, args, log):
    """Identity-aligned retriever. Anchors are (query, candidate) pairs, whose
    identity is the candidate's, and memory entries themselves."""
    opt = torch.optim.AdamW(R.parameters(), lr=args.lr, weight_decay=0.01)
    Q, N = c["cand"].shape
    pid_m, Mn = mem["pid"], len(mem["pid"])
    order = torch.argsort(pid_m)
    sorted_pid = pid_m[order]
    # Look-alike hard negatives: memory entries whose captions read most like
    # the query; per anchor we keep those showing a different person.
    look = torch.cat([(c["txt"][s:s + 2048].float() @ mem["txt"].float().T).topk(32, dim=1)[1]
                      for s in range(0, Q, 2048)])

    def random_of_pid(p):
        lo = torch.searchsorted(sorted_pid, p)
        hi = torch.searchsorted(sorted_pid, p, right=True)
        pick = lo + (torch.rand(len(p), device=dev) * (hi - lo)).long()
        return order[pick.clamp(max=Mn - 1)], hi > lo

    R.train()
    for step in range(args.steps_r):
        b = args.bs_r
        q = torch.randint(Q, (b,), device=dev)
        j = torch.randint(N, (b,), device=dev)
        cand = c["cand"][q, j]
        a_pid = c["gid"][cand]
        a_own = TGT * M.DOMAIN_STRIDE + q
        mi = torch.randint(Mn, (b,), device=dev)  # memory entries as anchors too
        pos_q, has = random_of_pid(a_pid)
        pos_m, _ = random_of_pid(pid_m[mi])
        hard = look[q, torch.randint(32, (b,), device=dev)]
        hard = hard[pid_m[hard] != a_pid]
        batch = torch.unique(torch.cat([pos_q[has], pos_m, hard, mi, torch.randint(Mn, (args.neg_r,), device=dev)]))
        za = torch.cat([R(c["cross"][q, j], c["txt"][q], c["img"][cand]),
                        R(mem["cross"][mi], mem["txt"][mi], mem["img"][mi])])
        zm = R(mem["cross"][batch], mem["txt"][batch], mem["img"][batch])
        pos = torch.cat([a_pid, pid_m[mi]])[:, None] == pid_m[batch][None]
        valid = torch.cat([a_own, mem["owner"][mi]])[:, None] != mem["owner"][batch][None]
        loss = alignment_loss(za, zm, pos, valid, args.tau)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps_r - 1:
            log(f"  R step {step:5d}  L_align {loss.item():.4f}")
    R.eval()


def train_scorer(g, c, mem, zm, zq, ev_idx, ev_sim, lam, args, log):
    opt = torch.optim.AdamW(g.parameters(), lr=args.lr, weight_decay=0.01)
    Q, N = c["cand"].shape
    k = g.k
    ev_idx, ev_sim = ev_idx[..., :k], ev_sim[..., :k]
    pos_all = c["gid"][c["cand"]] == c["qid"][:, None]
    rows = torch.nonzero(pos_all.any(1))[:, 0]
    log(f"  {len(rows)}/{Q} training queries have a match in their top-{N}")
    g.train()
    for step in range(args.steps_g):
        q = rows[torch.randint(len(rows), (args.bs_g,), device=dev)]
        B = len(q)
        p = pairs(c, q)
        ec, er = evidence(mem, ev_idx[q].reshape(B * N, k), ev_sim[q].reshape(B * N, k), p["txt"], p["img"])
        s_rel = p["base"][:, 0] + g(p["cross"], p["base"], ec, er)
        # Irrelevant context: random entries, sorted by key similarity so the
        # block order means the same as for retrieved evidence.
        ridx = torch.randint(len(mem["pid"]), (B * N, k), device=dev)
        rsim = (zq[q].reshape(B * N, 1, -1) * zm[ridx]).sum(-1)
        rsim, o = rsim.sort(1, descending=True)
        ec, er = evidence(mem, ridx.gather(1, o), rsim, p["txt"], p["img"])
        s_irr = p["base"][:, 0] + g(p["cross"], p["base"], ec, er)
        l_rel = ranking_loss(s_rel.view(B, N), pos_all[q])
        l_irr = ranking_loss(s_irr.view(B, N), pos_all[q])
        loss = l_rel + lam * adherence_loss(l_rel, l_irr, args.margin)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0 or step == args.steps_g - 1:
            log(f"  g step {step:5d}  L_rel {l_rel.item():.4f}  L_irrel {l_irr.item():.4f}")
    g.eval()


# -------------------------------------------------------------- evaluation

@torch.no_grad()
def metrics(c, s_top, chunk=512):
    """R1/R5/R10/mAP over the full gallery: the top-N candidates ordered by
    s_top come first, the rest of the gallery follows in ITC order."""
    s_top = torch.as_tensor(s_top, device=dev).float()
    img = c["img"].float()
    cmc, ap = [], []
    for s in range(0, len(c["txt"]), chunk):
        S = c["txt"][s:s + chunk].float() @ img.T
        top = s_top[s:s + chunk]
        S.scatter_(1, c["cand"][s:s + chunk], top - top.min(1, keepdim=True)[0] + 10.0)  # ITC cosine < 10
        match = c["gid"][S.argsort(1, descending=True)] == c["qid"][s:s + chunk, None]
        hits = match.cumsum(1)
        cmc.append(hits[:, :10] > 0)
        prec = hits / torch.arange(1, match.shape[1] + 1, device=dev)
        ap.append((prec * match).sum(1) / match.sum(1).clamp(min=1))
    cmc = torch.cat(cmc).float().mean(0) * 100
    return dict(R1=cmc[0].item(), R5=cmc[4].item(), R10=cmc[9].item(), mAP=torch.cat(ap).mean().item() * 100)


@torch.no_grad()
def score_with_g(g, c, mem, ev_idx, ev_sim, chunk=256):
    Q, N = c["cand"].shape
    k = g.k
    ev_idx, ev_sim = ev_idx[..., :k], ev_sim[..., :k]
    out = []
    for s in range(0, Q, chunk):
        q = torch.arange(s, min(Q, s + chunk), device=dev)
        B = len(q)
        p = pairs(c, q)
        ec, er = evidence(mem, ev_idx[q].reshape(B * N, k), ev_sim[q].reshape(B * N, k), p["txt"], p["img"])
        out.append((p["base"][:, 0] + g(p["cross"], p["base"], ec, er)).view(B, N))
    return torch.cat(out)


@torch.no_grad()
def raw_evidence(c, mem, k, chunk=512):
    """Evidence picked by raw ITC cosine instead of R (retriever-swap ablation,
    RA-VLA Sec. 5.4): key = normalised [caption; image] embedding."""
    Q, N = c["cand"].shape
    mkey = F.normalize(torch.cat([mem["txt"].float(), mem["img"].float()], 1), dim=-1)
    idx, sim = [], []
    for s in range(0, Q, chunk):
        q = torch.arange(s, min(Q, s + chunk), device=dev)
        B = len(q)
        key = F.normalize(torch.cat([c["txt"][q].float()[:, None].expand(B, N, -1),
                                     c["img"][c["cand"][q]].float()], -1), dim=-1).reshape(B * N, -1)
        i, v = retrieve(key, (TGT * M.DOMAIN_STRIDE + q).repeat_interleave(N), mkey, mem["owner"], k)
        idx.append(i.view(B, N, k))
        sim.append(v.view(B, N, k))
    return torch.cat(idx), torch.cat(sim)


@torch.no_grad()
def random_evidence(R, c, zm, k, chunk=512):
    """Random memory entries per pair, sorted by key similarity like retrieved ones."""
    Q, N = c["cand"].shape
    idx, sim = [], []
    for s in range(0, Q, chunk):
        q = torch.arange(s, min(Q, s + chunk), device=dev)
        p = pairs(c, q)
        zq = R(p["cross"], p["txt"], p["img"])
        i = torch.randint(len(zm), (len(zq), k), device=dev)
        v, o = (zq[:, None] * zm[i]).sum(-1).sort(1, descending=True)
        idx.append(i.gather(1, o).view(len(q), N, k))
        sim.append(v.view(len(q), N, k))
    return torch.cat(idx), torch.cat(sim)


def contextual_sensitivity(s_ret, s_rand):
    """RA-VLA eq. (8) on the candidate score vector of each query:
    ||s(D_ret) - s(D_rand)|| / ||s(D_ret)||, both centred per query so a
    constant shift (which leaves the ranking unchanged) does not count."""
    a = s_ret - s_ret.mean(1, keepdim=True)
    b = s_rand - s_rand.mean(1, keepdim=True)
    return ((a - b).norm(dim=1) / a.norm(dim=1).clamp(min=1e-6)).mean().item()


def timed(fn):
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    out = fn()
    if dev == "cuda":
        torch.cuda.synchronize()
    return out, time.time() - t


@torch.no_grad()
def training_free(c, mem, k, beta, kind, chunk=512):
    """Baselines without any learned module.
    knn:     evidence = memory pairs nearest to the (query, candidate) pair in
             raw ITC space; bonus = how well they agree with both sides.
    rocchio: evidence = memory captions nearest to the query; the query is
             moved toward their images (RA-VLA's informed initialisation)."""
    Q, N = c["cand"].shape
    mt, mi = mem["txt"].float(), mem["img"].float()
    mkey = F.normalize(torch.cat([mt, mi], 1), dim=-1)
    out = []
    for s in range(0, Q, chunk):
        q = torch.arange(s, min(Q, s + chunk), device=dev)
        B = len(q)
        tq, ic = c["txt"][q].float(), c["img"][c["cand"][q]].float()
        own = TGT * M.DOMAIN_STRIDE + q
        if kind == "knn":
            key = F.normalize(torch.cat([tq[:, None].expand(B, N, -1), ic], -1), dim=-1).reshape(B * N, -1)
            idx, _ = retrieve(key, own.repeat_interleave(N), mkey, mem["owner"], k)
            idx = idx.view(B, N, k)
            bonus = 0.5 * ((tq[:, None, None] * mt[idx]).sum(-1) + (ic[:, :, None] * mi[idx]).sum(-1)).mean(-1)
        else:
            idx, _ = retrieve(tq, own, mt, mem["owner"], k)
            qp = F.normalize(tq + mi[idx].mean(1), dim=-1)
            bonus = (qp[:, None] * ic).sum(-1)
        out.append(c["itm"][q] + beta * bonus)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source train cache (extracted with --gt)")
    ap.add_argument("--target", action="append", required=True, help="name=path of a target cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4, help="evidence entries per pair")
    ap.add_argument("--ks", default="1,2,4,8", help="K values for the sensitivity sweep (one scorer each)")
    ap.add_argument("--pseudo-fracs", default="0,0.25,0.5,1", help="fractions of target pseudo-pairs kept")
    ap.add_argument("--ntr", type=int, default=16, help="candidates per training query")
    ap.add_argument("--steps-r", type=int, default=3000)
    ap.add_argument("--steps-g", type=int, default=3000)
    ap.add_argument("--bs-r", type=int, default=512)
    ap.add_argument("--neg-r", type=int, default=2048)
    ap.add_argument("--bs-g", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the adherence loss")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--pseudo-r", type=int, default=5)
    ap.add_argument("--min-logodds", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"args: {vars(args)}")
    src = load(args.src)
    assert "gt_cross" in src, "source cache must be extracted with --gt"

    # -- simulated adaptation inside the source domain
    pids = np.unique(src["qid"])
    half = rng.permutation(pids)[: len(pids) // 2]
    inA = np.isin(src["qid"], half)
    qA, qB = np.nonzero(inA)[0], np.nonzero(~inA)[0]
    pos_in_B = np.full(len(src["qid"]), -1)
    pos_in_B[qB] = np.arange(len(qB))
    ps = M.pseudo(src, SRC, qB, r=args.pseudo_r, min_logodds=args.min_logodds)
    mem_np = M.concat(M.labelled(src, SRC, qA), ps)
    log("simulation " + M.describe(mem_np, lambda o: src["qid"][o % M.DOMAIN_STRIDE]))
    # B queries are ranked as TGT queries 0..|B|-1, so their pseudo entries are re-owned to match.
    ps_mask = mem_np["pseudo"] == 1
    mem_np["owner"][ps_mask] = M.owner_ids(TGT, pos_in_B[mem_np["owner"][ps_mask] % M.DOMAIN_STRIDE])
    mem = to_dev(mem_np)
    sim = gpu(src, qB, args.ntr)

    t0 = time.time()
    R = Retriever().to(dev)
    log("training retriever R")
    train_retriever(R, sim, mem, args, log)
    zm = encode_memory(R, mem)
    ks = sorted({int(k) for k in args.ks.split(",")} | {args.k})
    kmax = max(ks)
    ev_idx, ev_sim = evidence_for(R, sim, mem, zm, kmax)
    zq = keys(R, sim)
    same = (mem["pid"][ev_idx[..., :args.k]] == sim["gid"][sim["cand"]][..., None]).float()  # source labels
    log(f"  R: evidence shows the candidate's identity {100 * same.mean():.1f}% "
        f"(at least one of top-{args.k}: {100 * same.max(-1)[0].mean():.1f}%)")

    # "full" and "no_adhere" use K=args.k; "k<n>" are the K-sensitivity scorers.
    plan = [("full", args.k, args.lam), ("no_adhere", args.k, 0.0)]
    plan += [(f"k{k}", k, args.lam) for k in ks if k != args.k]
    scorers = {}
    for name, k, lam in plan:
        log(f"training scorer g [{name}, K={k}, lambda={lam}]")
        g = Scorer(k).to(dev)
        train_scorer(g, sim, mem, zm, zq, ev_idx, ev_sim, lam, args, log)
        scorers[name] = g
    scorers[f"k{args.k}"] = scorers["full"]
    log(f"training done in {time.time() - t0:.0f}s")

    # beta of the training-free baselines, tuned on the source simulation
    betas = {}
    for kind in ("knn", "rocchio"):
        r1, b = max((metrics(sim, training_free(sim, mem, args.k, b, kind))["R1"], b)
                    for b in (0.0, 1.0, 2.0, 4.0, 8.0, 16.0))
        betas[kind] = b
        log(f"  {kind}: beta={b} (simulation R1 {r1:.2f})")
    log(f"  simulation R1: ITM {metrics(sim, sim['itm'])['R1']:.2f}, "
        f"RA-TBPS {metrics(sim, score_with_g(scorers['full'], sim, mem, ev_idx, ev_sim))['R1']:.2f}")
    del sim, mem, zm, zq, ev_idx, ev_sim

    # -- test-time adaptation on each target: no weight updates
    results = {}
    mem_src = M.labelled(src, SRC)
    for spec in args.target:
        name, path = spec.split("=", 1)
        d = load(path)
        c = gpu(d)
        if dev == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        mem_t_np = M.concat(mem_src, M.pseudo(d, TGT, r=args.pseudo_r, min_logodds=args.min_logodds))
        log(f"[{name}] " + M.describe(mem_t_np, lambda o: d["qid"][o % M.DOMAIN_STRIDE]))
        mem_t_np["pid"] = np.where(mem_t_np["pseudo"] == 1, -1, mem_t_np["pid"])  # target labels never used
        mem_t = to_dev(mem_t_np)
        zm_t = encode_memory(R, mem_t)
        ev_i, ev_v = evidence_for(R, c, mem_t, zm_t, kmax)
        s_full = score_with_g(scorers["full"], c, mem_t, ev_i, ev_v)
        if dev == "cuda":
            torch.cuda.synchronize()
        adapt_time = time.time() - t1
        vram = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else 0.0
        Q = len(d["qid"])

        rows = {
            "ITC only": metrics(c, c["itc"]),
            "APTM (ITC->ITM)": metrics(c, c["itm"]),
            "+ kNN memory (training-free)": metrics(c, training_free(c, mem_t, args.k, betas["knn"], "knn")),
            "+ Q-prototype (training-free)": metrics(c, training_free(c, mem_t, args.k, betas["rocchio"], "rocchio")),
            "+ R + g, no adherence loss": metrics(c, score_with_g(scorers["no_adhere"], c, mem_t, ev_i, ev_v)),
            "+ R + g (RA-TBPS)": metrics(c, s_full),
        }
        # Contextual sensitivity (RA-VLA Table 3): swap retrieved for random evidence.
        ridx, rsim = random_evidence(R, c, zm_t, kmax)
        s_rand = score_with_g(scorers["full"], c, mem_t, ridx, rsim)
        sctx = {"RA-TBPS": contextual_sensitivity(s_full, s_rand),
                "no adherence loss": contextual_sensitivity(
                    score_with_g(scorers["no_adhere"], c, mem_t, ev_i, ev_v),
                    score_with_g(scorers["no_adhere"], c, mem_t, ridx, rsim))}
        rows["  diag: random evidence"] = metrics(c, s_rand)
        # Retriever swap (RA-VLA Sec. 5.4): same g, evidence from raw ITC cosine.
        rows["  diag: raw-cosine retriever"] = metrics(c, score_with_g(scorers["full"], c, mem_t,
                                                                       *raw_evidence(c, mem_t, kmax)))
        rows["  diag: source memory only"] = metrics(
            c, score_with_g(scorers["full"], c, mem_t, *evidence_for(R, c, mem_t, zm_t, kmax, mem_t["pseudo"] == 0)))
        if int(mem_t["pseudo"].sum()) > kmax:
            rows["  diag: pseudo memory only"] = metrics(
                c, score_with_g(scorers["full"], c, mem_t,
                                *evidence_for(R, c, mem_t, zm_t, kmax, mem_t["pseudo"] == 1)))

        # K sensitivity and scoring latency (RA-VLA Table 4 / Fig. 5)
        k_rows = {}
        for k in ks:
            s, sec = timed(lambda: score_with_g(scorers[f"k{k}"], c, mem_t, ev_i, ev_v))
            k_rows[k] = dict(metrics(c, s), ms_per_query=1000 * sec / Q)
        # Buffer size N: fraction of the target pseudo-pairs kept (source part fixed)
        n_rows = {}
        gen = torch.Generator(device=dev).manual_seed(args.seed)
        coin = torch.rand(len(mem_t["pid"]), generator=gen, device=dev)
        for frac in (float(f) for f in args.pseudo_fracs.split(",")):
            keep = (mem_t["pseudo"] == 0) | (coin < frac)
            n_rows[frac] = dict(metrics(c, score_with_g(scorers["full"], c, mem_t,
                                                        *evidence_for(R, c, mem_t, zm_t, kmax, keep))),
                                pseudo=int((keep & (mem_t["pseudo"] == 1)).sum()))

        log(f"[{name}] adaptation {adapt_time:.1f}s for {Q} queries, peak VRAM {vram:.2f} GiB")
        log(f"{'method':34s}{'R1':>8s}{'R5':>8s}{'R10':>8s}{'mAP':>8s}")
        for k_, m in rows.items():
            log(f"{k_:34s}{m['R1']:8.2f}{m['R5']:8.2f}{m['R10']:8.2f}{m['mAP']:8.2f}")
        log("contextual sensitivity S_ctx: " + ", ".join(f"{k_} {v:.4f}" for k_, v in sctx.items()))
        log(f"{'K':>4s}{'R1':>8s}{'mAP':>8s}{'ms/query':>10s}")
        for k, m in k_rows.items():
            log(f"{k:4d}{m['R1']:8.2f}{m['mAP']:8.2f}{m['ms_per_query']:10.3f}")
        log(f"{'pseudo kept':>12s}{'entries':>9s}{'R1':>8s}{'mAP':>8s}")
        for f_, m in n_rows.items():
            log(f"{f_:12.2f}{m['pseudo']:9d}{m['R1']:8.2f}{m['mAP']:8.2f}")
        results[name] = dict(rows=rows, s_ctx=sctx, k_sweep=k_rows, n_sweep=n_rows,
                             adapt_seconds=adapt_time, peak_vram_gib=vram,
                             memory=dict(entries=len(mem_t_np["pid"]), pseudo=int(mem_t_np["pseudo"].sum())))
        del c, mem_t, zm_t, ev_i, ev_v

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(dict(args=vars(args), betas=betas, results=results), f, indent=1)
    torch.save({"R": R.state_dict(), **{f"g_{k}": g.state_dict() for k, g in scorers.items()}},
               os.path.join(args.out, "modules.pt"))
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
