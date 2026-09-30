from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import get_context


class Qwen3_5RMSNorm(nn.Module):
    """RMSNorm with the Qwen3.5 formula: rms(x) * (1 + weight), weight init to 0."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        h = h.to(input_dtype) * (1.0 + self.weight)
        return h.type_as(x)


class Qwen3_5RMSNormGated(nn.Module):
    """rms(x) * weight * silu(gate); weight init to ones."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        h = self.weight * h.to(input_dtype)
        h = h * F.silu(gate.to(torch.float32))
        return h.to(input_dtype)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


@lru_cache(1)
def _partial_rope_inv_freq(rotary_dim: int, base: float) -> torch.Tensor:
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    return inv_freq


class Qwen3_5PartialRotary(nn.Module):
    """Text M-RoPE reduced to single-dim rotation on the first `rotary_dim` dims."""

    def __init__(self, rotary_dim: int, base: float) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        inv_freq = _partial_rope_inv_freq(rotary_dim, base)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("i,j->ij", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(self, positions, query, key):
        rotary_dim = self.rotary_dim
        cos, sin = self._cos_sin(positions)
        cos = cos.to(query.dtype).unsqueeze(1)
        sin = sin.to(query.dtype).unsqueeze(1)
        q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
        k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
        q_rot = q_rot * cos + _rotate_half(q_rot) * sin
        k_rot = k_rot * cos + _rotate_half(k_rot) * sin
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


def _torch_chunk_gated_delta_rule(
    query, key, value, g, beta, initial_state=None, chunk_size=64,
):
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:2], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length].transpose(1, 2)
    return core_attn_out, last_recurrent_state


def _torch_recurrent_gated_delta_rule(query, key, value, g, beta, initial_state=None):
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, dtype=value.dtype, device=value.device)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    core_attn_out = core_attn_out.transpose(1, 2)
    return core_attn_out, last_recurrent_state


class GatedDeltaNet(nn.Module):
    """Linear-attention (GatedDeltaNet) token mixer. Linear layers hold recurrent state in ModelRunner slots."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        conv_dim = self.key_dim * 2 + self.value_dim

        self.conv1d = nn.Conv1d(conv_dim, conv_dim, bias=False,
                                kernel_size=config.linear_conv_kernel_dim, groups=conv_dim)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.log(torch.empty(self.num_v_heads).uniform_(0.01, 16.0)))

        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=config.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, config.hidden_size, bias=False)

        self.in_proj_qkv = nn.Linear(config.hidden_size, self.key_dim * 2 + self.value_dim, bias=False)
        self.in_proj_z = nn.Linear(config.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(config.hidden_size, self.num_v_heads, bias=False)

        # allocated by ModelRunner
        self.recurrent_state = torch.tensor([])
        self.conv_state = torch.tensor([])
        # per-draft-position state snapshots for speculative verification
        self.verify_snap_rec = torch.tensor([])
        self.verify_snap_conv = torch.tensor([])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        context = get_context()
        has_state = self.recurrent_state.numel() > 0
        kernel = self.conv1d.weight.shape[-1]
        state_len = kernel - 1

        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states).sigmoid()
        a = self.in_proj_a(hidden_states)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        raw = self.in_proj_qkv(hidden_states)
        slots = context.seq_slots

        outputs = []
        if context.is_prefill:
            starts = context.cu_seqlens_q[:-1]
            ends = context.cu_seqlens_q[1:]
            if context.is_verify:
                # Speculative verify: each segment holds the K draft tokens. Step
                # through them one token at a time and snapshot the state after
                # each, so the runner can roll back to any accepted prefix.
                for i in range(len(starts)):
                    s, e = starts[i].item(), ends[i].item()
                    slot = slots[s].item() if slots is not None else 0
                    conv_run = self.conv_state[slot]                  # [conv_dim, state_len]
                    rec = self.recurrent_state[slot].unsqueeze(0)     # [1, H, kdim, vdim]
                    for t in range(s, e):
                        conv_in = torch.cat([conv_run, raw[t:t + 1].transpose(0, 1)], dim=1)
                        convi = F.conv1d(conv_in[None], self.conv1d.weight,
                                         groups=self.conv1d.groups)[0, :, -1]  # [conv_dim]
                        conv_run = conv_in[:, -state_len:].contiguous()
                        mixed = F.silu(convi).view(1, -1)             # [1, conv_dim]
                        query_t, key_t, value_t = torch.split(
                            mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                        q_t = l2norm(query_t.reshape(1, self.num_v_heads, self.head_k_dim), eps=1e-6)
                        k_t = l2norm(key_t.reshape(1, self.num_k_heads, self.head_k_dim), eps=1e-6)
                        if self.num_v_heads // self.num_k_heads > 1:
                            k_t = k_t.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
                        v_t = value_t.reshape(1, self.num_v_heads, self.head_v_dim)
                        out_t, rec = _torch_recurrent_gated_delta_rule(
                            q_t.unsqueeze(2).float(), k_t.unsqueeze(2).float(), v_t.unsqueeze(2).float(),
                            g[t].view(1, -1, 1).float(), b[t].view(1, -1, 1).float(), rec)
                        self.verify_snap_rec[t - s, slot].copy_(rec[0])
                        self.verify_snap_conv[t - s, slot].copy_(conv_run)
                        outputs.append(out_t.squeeze(2).squeeze(0).to(hidden_states.dtype))
                    if has_state:
                        self.recurrent_state[slot].copy_(rec[0])
                        self.conv_state[slot].copy_(conv_run)
            else:
                for i in range(len(starts)):
                    s, e = starts[i].item(), ends[i].item()
                    slot = slots[s].item() if slots is not None else 0
                    raw_i = raw[s:e]  # [L, conv_dim]
                    L = e - s
                    if has_state:
                        # continue from stored conv state: prepend it so the conv
                        # window sees the previous state_len inputs (zero state on
                        # first chunk is equivalent to zero padding)
                        conv_in = torch.cat([self.conv_state[slot], raw_i.transpose(0, 1)], dim=1)  # [conv_dim, state_len + L]
                        convi = F.conv1d(conv_in[None], self.conv1d.weight,
                                         groups=self.conv1d.groups)[0]  # [conv_dim, L]
                        self.conv_state[slot].copy_(conv_in[:, -state_len:].contiguous())
                    else:
                        convi = F.conv1d(raw_i.transpose(0, 1)[None], self.conv1d.weight,
                                         padding=state_len, groups=self.conv1d.groups)[0, :, :L]
                        tail = raw_i.transpose(0, 1).contiguous()          # [conv_dim, L]
                        tail = F.pad(tail, (max(0, state_len - L), 0))     # left-zero pad to state_len
                        self.conv_state[slot].copy_(tail[:, -state_len:].contiguous())
                    mixed = F.silu(convi).transpose(0, 1)  # [L, conv_dim]
                    query_i, key_i, value_i = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
                    g_i = g[s:e]
                    b_i = b[s:e]
                    q_i = l2norm(query_i.reshape(-1, self.num_v_heads, self.head_k_dim), eps=1e-6)
                    k_i = l2norm(key_i.reshape(-1, self.num_k_heads, self.head_k_dim), eps=1e-6)
                    if self.num_v_heads // self.num_k_heads > 1:
                        k_i = k_i.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
                    v_i = value_i.reshape(-1, self.num_v_heads, self.head_v_dim)
                    init = self.recurrent_state[slot].unsqueeze(0) if has_state else None
                    out_1, state = _torch_chunk_gated_delta_rule(
                        q_i.transpose(0, 1).unsqueeze(0).float(),
                        k_i.transpose(0, 1).unsqueeze(0).float(),
                        v_i.transpose(0, 1).unsqueeze(0).float(),
                        g_i.T.unsqueeze(0).float(),
                        b_i.T.unsqueeze(0).float(),
                        init,
                    )
                    if has_state:
                        self.recurrent_state[slot].copy_(state[0])
                    # out_1: [1, L, H, vdim] -> token-major [L, H, vdim]
                    outputs.append(out_1.squeeze(0).to(hidden_states.dtype).contiguous())
            core_attn_out = torch.cat(outputs, dim=0).reshape(-1, self.head_v_dim)
        else:
            cur = raw.unsqueeze(2)  # [B, conv_dim, 1]
            if has_state:
                combined = torch.cat([self.conv_state[slots], cur], dim=2)  # [B, conv_dim, kernel]
            else:
                combined = cur
            conv = F.conv1d(combined, self.conv1d.weight, padding=0, groups=self.conv1d.groups)  # [B, conv_dim, 1]
            mixed = F.silu(conv.transpose(1, 2))  # [B, 1, conv_dim]
            if has_state:
                self.conv_state[slots] = combined[:, :, -state_len:].contiguous()
            query, key, value = torch.split(mixed, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q_i = l2norm(query.reshape(-1, self.num_v_heads, self.head_k_dim), eps=1e-6)
            k_i = l2norm(key.reshape(-1, self.num_k_heads, self.head_k_dim), eps=1e-6)
            if self.num_v_heads // self.num_k_heads > 1:
                k_i = k_i.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=1)
            v_i = value.reshape(-1, self.num_v_heads, self.head_v_dim)
            init_r = self.recurrent_state[slots] if has_state else None  # [B,H,kdim,vdim]
            q_r = q_i.transpose(0, 1).unsqueeze(2).transpose(0, 1)  # [B,H,1,kdim]
            k_r = k_i.transpose(0, 1).unsqueeze(2).transpose(0, 1)
            v_r = v_i.transpose(0, 1).unsqueeze(2).transpose(0, 1)
            g_r = g.unsqueeze(-1)  # [B,H,1]
            b_r = b.unsqueeze(-1)  # [B,H,1]
            out_1, state = _torch_recurrent_gated_delta_rule(
                q_r.float(), k_r.float(), v_r.float(), g_r.float(), b_r.float(), init_r)
            if has_state:
                self.recurrent_state[slots] = state  # [B,H,kdim,vdim]
            core_attn_out = out_1.squeeze(2).to(hidden_states.dtype).reshape(-1, self.head_v_dim)

        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(-1, self.value_dim)
        return self.out_proj(core_attn_out)


class Qwen3_5Attention(nn.Module):
    """Full-attention token mixer with q/k RMSNorm, partial rotary and gated output."""

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        rope_cfg = config.rope_parameters
        partial = rope_cfg.get("partial_rotary_factor", 1.0)
        rotary_dim = int(self.head_dim * partial)
        self.rotary = Qwen3_5PartialRotary(rotary_dim, rope_cfg.get("rope_theta", 10000000.0))

        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape[:-1]
        q_gate = self.q_proj(hidden_states).view(*shape, -1, self.head_dim * 2)
        query_states, gate = torch.chunk(q_gate, 2, dim=-1)
        gate = gate.reshape(*shape, -1)

        query_states = self.q_norm(query_states.reshape(*shape, -1, self.head_dim))
        key_states = self.k_norm(self.k_proj(hidden_states).view(*shape, -1, self.head_dim))
        value_states = self.v_proj(hidden_states).view(*shape, -1, self.head_dim)

        query_states, key_states = self.rotary(positions, query_states, key_states)

        attn_output = self.attn(query_states, key_states, value_states)
        attn_output = attn_output.reshape(*shape, -1) * torch.sigmoid(gate)
        return self.o_proj(attn_output)


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, layer_type: str | None = None):
        super().__init__()
        self.block_type = layer_type or config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        else:
            raise ValueError(f"Unsupported layer type: {self.block_type}")
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            h = self.linear_attn(h)
        else:
            h = self.self_attn(h, positions)
        hidden_states = residual + h

        residual = hidden_states
        h = self.post_attention_layernorm(hidden_states)
        h = self.mlp(h)
        hidden_states = residual + h
        return hidden_states


class Qwen3_5Model(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5MTP(nn.Module):
    """EAGLE-style multi-token-prediction drafter (shares embedding / lm_head with the target).

    At sequence position i the drafter consumes the pair (token t_i, target hidden h_{i-1})
    — exactly the MTP training convention — and its logits at position i predict t_{i+1}.
    """

    def __init__(self, config: Qwen3_5TextConfig, embed_tokens: VocabParallelEmbedding):
        super().__init__()
        self.embed_tokens = embed_tokens    # shared with the target model
        self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, i, layer_type="full_attention")
            for i in range(getattr(config, "mtp_num_hidden_layers", 0))
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, target_hidden: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.pre_fc_norm_embedding(self.embed_tokens(input_ids))
        target_hidden = self.pre_fc_norm_hidden(target_hidden)
        hidden_states = self.fc(torch.cat([inputs_embeds, target_hidden], dim=-1))
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    weight_prefix = "model.language_model."

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.embed_tokens.weight.data
        else:
            raise ValueError("Qwen3.5-0.8B requires tied word embeddings")
        if getattr(config, "mtp_num_hidden_layers", 0):
            self.mtp = Qwen3_5MTP(config, self.embed_tokens)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)