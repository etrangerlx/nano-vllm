import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer
from transformers import Qwen3_5ForConditionalGeneration
from safetensors import safe_open

sys.path.insert(0, "e:/project/nano-vllm")
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM, GatedDeltaNet
from nanovllm.utils.loader import load_model, default_weight_loader
from nanovllm.utils.context import set_context, reset_context
from nanovllm.layers.attention import Attention

BLOCK = 512

MODEL_PATH = "E:/models/Qwen3.5-0.8B"
PROMEMTS = [
    "介绍一下你自己",
    "hello world",
    "1 + 1 = ",
]


_BUF = {}
def attach_states(mine, tc):
    """Allocate one persistent linear-state slot per layer (as ModelRunner does)."""
    num_linear = sum(1 for m in mine.modules() if isinstance(m, GatedDeltaNet))
    v_heads = tc.linear_num_value_heads
    k_dim = tc.linear_key_head_dim
    v_dim = tc.linear_value_head_dim
    k_heads = tc.linear_num_key_heads
    conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
    state_len = tc.linear_conv_kernel_dim - 1
    rstate = torch.zeros(num_linear, 1, v_heads, k_dim, v_dim, dtype=torch.float32)
    cstate = torch.zeros(num_linear, 1, conv_dim, state_len, dtype=tc.dtype)
    li = 0
    for m in mine.modules():
        if isinstance(m, GatedDeltaNet):
            m.recurrent_state = rstate[li]
            m.conv_state = cstate[li]
            li += 1
    _BUF["rstate"], _BUF["cstate"] = rstate, cstate

    # full-attention kv cache: 2 heads, enough blocks for up to 512 tokens
    num_full = sum(1 for m in mine.modules() if isinstance(m, Attention))
    kv = torch.zeros(2, num_full, 32, BLOCK, tc.num_key_value_heads, tc.head_dim, dtype=tc.dtype)
    ai = 0
    for m in mine.modules():
        if isinstance(m, Attention):
            m.k_cache = kv[0, ai]
            m.v_cache = kv[1, ai]
            ai += 1
    _BUF["kv"] = kv


def reset_states():
    _BUF["rstate"].zero_()
    _BUF["cstate"].zero_()
    _BUF["kv"].zero_()


def prefill(mine, ids):
    T = ids.shape[0]
    set_context(True, torch.tensor([0, T], dtype=torch.int32),
                torch.tensor([0, T], dtype=torch.int32), T, T,
                slot_mapping=torch.arange(T, dtype=torch.int32),    # store kv into block 0
                seq_slots=torch.tensor([0] * T, dtype=torch.int32))
    return mine(ids, torch.arange(T))


def decode_step(mine, token, pos):
    block = pos // BLOCK
    row = pos % BLOCK
    set_context(False, slot_mapping=torch.tensor([row], dtype=torch.int32),
                context_lens=torch.tensor([pos + 1], dtype=torch.int32),
                block_tables=torch.tensor([[block]], dtype=torch.int32),
                seq_slots=torch.tensor([0], dtype=torch.int32))
    h = mine(torch.tensor([token]), torch.tensor([pos]))
    return mine.lm_head(h)[0]


def compare_decode(mine, ref_full, tok, text, n_decode=2):
    """Prefill into state slot, decode n tokens statefully; compare vs reference recompute."""
    enc = tok(text, return_tensors="pt")
    prompt = enc["input_ids"].tolist()[0]
    T0 = len(prompt)

    reset_states()
    prefill(mine, torch.tensor(prompt))          # seed chunk/recurrent/conv state

    full_ids = prompt.copy()
    print(f"\n[{text}] prefill T={T0}")
    for step in range(n_decode):
        # next token (at position len(full_ids)) per reference, from scratch
        r_out = ref_full(torch.tensor([full_ids]))
        nxt = int(r_out[0, -1].argmax())
        full_ids.append(nxt)                       # now includes the appended token

        # reference prediction for the token AFTER the appended one
        ref_logits = ref_full(torch.tensor([full_ids]))[0, -1]
        ref_top = int(ref_logits.argmax())

        # stateful decode of the appended token at position len(full_ids)-1
        reset_context()
        m_logits = decode_step(mine, nxt, len(full_ids) - 1)
        m_top = int(m_logits.argmax())

        # sanity: my own full recompute (chunk-gated) vs recurrent decode
        reset_context()
        recomp_h = prefill_mode_recomp(mine, torch.tensor(full_ids))
        self_recomp = mine.lm_head(recomp_h)[-1]
        dself = (self_recomp.float() - m_logits.float()).abs().max().item()

        d = (ref_logits.float() - m_logits.float()).abs().max().item()
        print(f"  step{step}: ref_top={ref_top} mine_decode={m_top} "
              f"max|d-vs-ref|={d:.4f} max|d-self-recomp|={dself:.4f} match={ref_top==m_top}")


def prefill_mode_recomp(mine, ids):
    """Same as prefill but avoids going through recurrent_state buffers."""
    # temporarily detach states so chunk path uses fresh-state mode, then restore values
    T = ids.shape[0]
    set_context(True, torch.tensor([0, T], dtype=torch.int32),
                torch.tensor([0, T], dtype=torch.int32), T, T,
                slot_mapping=torch.arange(T, dtype=torch.int32),
                seq_slots=torch.tensor([0] * T, dtype=torch.int32))
    saved_rec = {}
    saved_conv = {}
    for m in mine.modules():
        if isinstance(m, GatedDeltaNet):
            saved_rec[id(m)] = m.recurrent_state
            saved_conv[id(m)] = m.conv_state
            m.recurrent_state = torch.tensor([])
            m.conv_state = torch.tensor([])
    out = mine(ids, torch.arange(T))
    for m in mine.modules():
        if isinstance(m, GatedDeltaNet):
            m.recurrent_state = saved_rec[id(m)]
            m.conv_state = saved_conv[id(m)]
    return out


def main():
    torch.set_grad_enabled(False)
    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    tc = cfg.text_config
    torch.set_default_dtype(tc.dtype)

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

    ref = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=tc.dtype)
    ref.eval()
    def ref_full(ids):
        with torch.no_grad():
            return ref(input_ids=ids, return_dict=True).logits  # [B,T,V]

    mine = Qwen3_5ForCausalLM(tc)
    load_model(mine, MODEL_PATH)
    mine.eval()
    attach_states(mine, tc)

    print("params", len(list(mine.parameters())))

    for text in PROMEMTS:
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"]
        r_last = ref(input_ids=input_ids, return_dict=True).logits[0, -1]
        ids1d = input_ids[0]
        T = ids1d.shape[0]
        reset_states()
        hidden = prefill(mine, ids1d)
        m_last = mine.lm_head(hidden)[-1]
        reset_context()
        maxabs = (r_last.float() - m_last.float()).abs().max().item()
        print(f"\n[PREFILL {text}] T={T} max|d|={maxabs:.4f} "
              f"top1 ref={int(r_last.argmax())} mine={int(m_last.argmax())} "
              f"match={int(r_last.argmax())==int(m_last.argmax())}")

    # decode validation
    compare_decode(mine, ref_full, tok, "介绍一下你自己", n_decode=2)
    compare_decode(mine, ref_full, tok, "1 + 1 = ", n_decode=3)


if __name__ == "__main__":
    main()