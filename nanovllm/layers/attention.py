import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    valid_mask = slot_mapping != -1
    valid_slots = slot_mapping[valid_mask].long()
    if valid_slots.numel():
        # Flatten block/binslot dimension into a single slot dimension
        total_blocks, block_size = k_cache.shape[:2]
        k_cache_flat = k_cache.view(total_blocks * block_size, *k_cache.shape[2:])
        v_cache_flat = v_cache.view(total_blocks * block_size, *v_cache.shape[2:])
        k_cache_flat.index_copy_(0, valid_slots, key[valid_mask])
        v_cache_flat.index_copy_(0, valid_slots, value[valid_mask])


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    @staticmethod
    def _scatter_to_padded(x, lengths, max_len):
        """把紧凑排列的 x 展开成带 padding 的批次张量。

        输入: x 形状 [sum(lengths), *rest_dims] —— 所有序列按顺序拼接
        输出: out 形状 [B, max_len, *rest_dims] —— 每个序列独立一行，不足补零

        示例: lengths=[3,2], x=[A,B,C,X,Y]
              → out[0]=[A,B,C,0], out[1]=[X,Y,0,0]
        """
        batch_size = lengths.shape[0]
        # 先创建一个全零的填充张量
        out = x.new_zeros(batch_size, max_len, *x.shape[1:])
        # 用切片方式，逐序列拷贝数据，直观易懂
        offset = 0
        for i in range(batch_size):
            seq_len = lengths[i].item()
            # 从紧凑 x 中取出第 i 条序列，长度为 seq_len
            seq_data = x[offset : offset + seq_len]
            # 写入填充张量第 i 行的前 seq_len 个位置
            out[i, :seq_len] = seq_data
            offset += seq_len
        return out

    @staticmethod
    def _gather_from_padded(x_padded, lengths):
        """把带 padding 的批次张量中的有效 token 收集起来，紧凑拼接。

        输入: x_padded 形状 [B, max_len, *rest_dims]
        输出: 形状 [sum(lengths), *rest_dims] —— 按 batch 顺序依次拼接有效部分

        示例: lengths=[3,2], x_padded=[[A,B,C,0],[X,Y,0,0]]
              → [A,B,C,X,Y]
        """
        # 收集每一条序列的有效部分（去掉 padding），然后按顺序拼接
        pieces = []
        for i, seq_len in enumerate(lengths.tolist()):
            # 取第 i 条序列的前 seq_len 个真实 token（丢掉后面的 padding）
            pieces.append(x_padded[i, :seq_len])
        return torch.cat(pieces, dim=0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            batch_size = context.cu_seqlens_q.size(0) - 1
            q_len = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
            k_len = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
            max_seqlen_q = context.max_seqlen_q
            max_seqlen_k = context.max_seqlen_k

            q_pad = self._scatter_to_padded(q, q_len, max_seqlen_q)

            if context.block_tables is not None:    # prefix cache
                safe_tables = context.block_tables.clamp(min=0)
                max_len = max_seqlen_k
                k = k_cache[safe_tables].view(batch_size, -1, self.num_kv_heads, self.head_dim)[:, :max_len]
                v = v_cache[safe_tables].view(batch_size, -1, self.num_kv_heads, self.head_dim)[:, :max_len]
            else:
                k_pad = self._scatter_to_padded(k, k_len, max_seqlen_k)
                v_pad = self._scatter_to_padded(v, k_len, max_seqlen_k)
                k, v = k_pad, v_pad

            # SDPA format: [B, H, T, D]
            q_sdpa = q_pad.transpose(1, 2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            if self.num_heads != self.num_kv_heads:
                num_group = self.num_heads // self.num_kv_heads
                k_sdpa = k_sdpa.repeat_interleave(num_group, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(num_group, dim=1)

            # Build float mask where -inf = masked out
            offset = k_len - q_len
            q_idx = torch.arange(max_seqlen_q, device=q.device).view(1, 1, -1, 1)
            k_idx = torch.arange(max_seqlen_k, device=q.device).view(1, 1, 1, -1)
            q_pad_mask = q_idx >= q_len.view(-1, 1, 1, 1)
            k_pad_mask = k_idx >= k_len.view(-1, 1, 1, 1)
            causal_mask = k_idx > (offset.view(-1, 1, 1, 1) + q_idx)
            mask = torch.zeros(batch_size, 1, max_seqlen_q, max_seqlen_k, dtype=q.dtype, device=q.device)
            mask.masked_fill_(q_pad_mask | k_pad_mask | causal_mask, float('-inf'))

            o_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask, scale=self.scale)
            o_pad = o_sdpa.transpose(1, 2)
            o = self._gather_from_padded(o_pad, q_len)

        else:    # decode
            batch_size = q.shape[0]
            max_len = context.context_lens.max().item()

            safe_tables = context.block_tables.clamp(min=0)
            k = k_cache[safe_tables].view(batch_size, -1, self.num_kv_heads, self.head_dim)[:, :max_len]
            v = v_cache[safe_tables].view(batch_size, -1, self.num_kv_heads, self.head_dim)[:, :max_len]

            q_sdpa = q.unsqueeze(2)
            k_sdpa = k.transpose(1, 2)
            v_sdpa = v.transpose(1, 2)
            if self.num_heads != self.num_kv_heads:
                num_group = self.num_heads // self.num_kv_heads
                k_sdpa = k_sdpa.repeat_interleave(num_group, dim=1)
                v_sdpa = v_sdpa.repeat_interleave(num_group, dim=1)

            mask_bool = (torch.arange(max_len, device=q.device).unsqueeze(0) >= context.context_lens.unsqueeze(1)).unsqueeze(1).unsqueeze(2)
            mask = torch.zeros(batch_size, 1, 1, max_len, dtype=q.dtype, device=q.device)
            mask.masked_fill_(mask_bool, float('-inf'))

            o_sdpa = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask, scale=self.scale)
            o = o_sdpa.squeeze(2)

        return o
