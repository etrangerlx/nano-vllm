import torch
from torch import nn


def _gumbel_sample(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float()
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        return _gumbel_sample(probs)

    @staticmethod
    def spec_verify(target_logits: torch.Tensor, draft_logits: torch.Tensor,
                    temperatures: torch.Tensor, draft_tokens: torch.Tensor):
        """Chain verification of K draft tokens (SpecDec rejection sampling).

        target_logits: [N, K+1, V] — target dists for positions p+1 .. p+K+1.
        draft_logits:  [N, K, V]   — drafter dists the draft tokens were sampled from.
        draft_tokens:  [N, K]
        Returns (num_accepted [N] long, corrected [N], bonus [N]). corrected is the
        resampled token at the first rejection (meaningful iff num_accepted < K);
        bonus is the +1 token sampled from the last target dist on full acceptance.
        """
        n, k = draft_tokens.shape
        t = temperatures.unsqueeze(1)
        active = torch.ones(n, dtype=torch.bool)
        num_accepted = torch.zeros(n, dtype=torch.long)
        corrected = torch.zeros(n, dtype=torch.long)
        for i in range(k):
            # [:, i] is the target dist for position p+i, produced by the token at
            # p+i-1 — it is the distribution draft_tokens[:, i] must be tested against
            p = torch.softmax(target_logits[:, i].float() / t, dim=-1)
            q = torch.softmax(draft_logits[:, i].float() / t, dim=-1)
            d = draft_tokens[:, i].unsqueeze(1)
            ratio = p.gather(1, d) / q.gather(1, d).clamp_min_(1e-10)
            acc = (torch.rand_like(ratio) < ratio).squeeze(1)      # implicit min(1, ·)
            rej = active & ~acc
            if rej.any():
                diff = (p - q).clamp_min_(0)
                empty = diff.sum(dim=-1, keepdim=True) < 1e-6
                corr = _gumbel_sample(torch.where(empty, p, diff))
                corrected[rej] = corr[rej]
            active = active & acc
            num_accepted += active.long()
            if not active.any():
                break
        bonus = _gumbel_sample(torch.softmax(target_logits[:, k].float() / t, dim=-1))
        return num_accepted, corrected, bonus
