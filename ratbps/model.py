"""Retriever R and grounded scorer g, with the two RA-VLA objectives.

R  (Sec. 4.2 of RA-VLA): a 2-layer Transformer over the pair's input tokens
   (fused state, caption embedding, image embedding), average-pooled into a
   key z. Trained with an identity-alignment contrastive loss so that pairs
   showing the same person land together, while look-alikes (similar captions,
   different person) are pushed apart as hard negatives.
g  (Sec. 4.3): refines the frozen ITM score of a (query, candidate) pair with
   K retrieved evidence entries. Pairwise grounding: block l attends to the
   pair state plus ONE evidence token, the most relevant entry in the last
   block. Trained with a listwise ranking loss plus the contextual adherence
   margin, which penalises g when random evidence scores as well as retrieved
   evidence (i.e. when g ignores its context).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_REL = 6  # relation features per evidence entry, see relation_features()


class Retriever(nn.Module):
    def __init__(self, d=256, out=128, layers=2, heads=4, cross_dim=768, emb_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(cross_dim, d), nn.Linear(emb_dim, d), nn.Linear(emb_dim, d)])
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d, out)

    def forward(self, cross, txt, img):
        tokens = torch.stack([p(x.float()) for p, x in zip(self.proj, (cross, txt, img))], 1) + self.type_emb
        return F.normalize(self.head(self.enc(tokens).mean(1)), dim=-1)


class GroundingBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, h, e):
        # h: (P, 1, d) pair state, e: (P, 1, d) one evidence token
        kv = self.n1(torch.cat([h, e], 1))
        h = h + self.attn(self.n1(h), kv, kv, need_weights=False)[0]
        return h + self.ff(self.n2(h))


class Scorer(nn.Module):
    def __init__(self, k, d=256, heads=4, cross_dim=768):
        super().__init__()
        self.k = k
        self.pair = nn.Linear(cross_dim, d)
        self.base = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.mem = nn.Linear(cross_dim, d)
        self.rel = nn.Sequential(nn.Linear(N_REL, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([GroundingBlock(d, heads) for _ in range(k)])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        nn.init.zeros_(self.out[1].weight)  # start as the frozen ITM ranking
        nn.init.zeros_(self.out[1].bias)

    def forward(self, cross, base, ev_cross, ev_rel):
        """cross (P,768), base (P,2)=[itm, itc], ev_cross (P,K,768) and
        ev_rel (P,K,N_REL) sorted most relevant first. Returns delta (P,)."""
        h = (self.pair(cross.float()) + self.base(base)).unsqueeze(1)
        e = self.mem(ev_cross.float()) + self.rel(ev_rel)
        for l, block in enumerate(self.blocks):
            h = block(h, e[:, self.k - 1 - l].unsqueeze(1))  # least relevant first
        return self.out(h[:, 0]).squeeze(-1)


def relation_features(tq, ic, ev_txt, ev_img, ev_zsim, ev_pseudo):
    """How evidence relates to the pair: does the evidence caption read like
    the query, does the evidence image look like the candidate, and the two
    cross terms. tq/ic (P,256); ev_* (P,K,...)."""
    tq, ic = tq.float().unsqueeze(1), ic.float().unsqueeze(1)
    ev_txt, ev_img = ev_txt.float(), ev_img.float()
    return torch.stack([(tq * ev_txt).sum(-1), (ic * ev_img).sum(-1), (tq * ev_img).sum(-1),
                        (ic * ev_txt).sum(-1), ev_zsim.float(), ev_pseudo.float()], -1)


def alignment_loss(za, zm, pos, valid, tau=0.07):
    """RA-VLA eq. (4) with identity positives: every positive (i, j) is
    contrasted against anchor i's negatives. za (A,d), zm (M,d), pos/valid (A,M)
    bool; valid masks out each anchor's own entry."""
    logits = za @ zm.T / tau
    neg = valid & ~pos
    neg_lse = torch.logsumexp(logits.masked_fill(~neg, -1e4), 1, keepdim=True)
    per_pos = -(logits - torch.logaddexp(logits, neg_lse))
    pos = pos & valid
    return (per_pos * pos).sum() / pos.sum().clamp(min=1)


def ranking_loss(s, pos):
    """Listwise: -log of the softmax mass on the correct candidates. s, pos (B,N)."""
    has = pos.any(1)
    lse_pos = torch.logsumexp(s.masked_fill(~pos, -1e4), 1)
    return (torch.logsumexp(s, 1) - lse_pos)[has].mean()


def adherence_loss(l_rel, l_irrel, margin):
    """RA-VLA eq. (6): max(0, m - (L_irrel - L_rel))."""
    return F.relu(margin - (l_irrel - l_rel))
