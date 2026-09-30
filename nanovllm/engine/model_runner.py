import pickle
import torch
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM, GatedDeltaNet
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def build_model(hf_config):
    if hf_config.model_type == "qwen2":
        return Qwen2ForCausalLM(hf_config)
    if hf_config.model_type == "qwen3":
        return Qwen3ForCausalLM(hf_config)
    if hf_config.model_type == "qwen3_5":
        return Qwen3_5ForCausalLM(getattr(hf_config, "text_config", hf_config))
    raise ValueError(f"Unsupported model_type: {hf_config.model_type}")


class ModelRunner:

    def __init__(self, config: Config, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.event = event

        self.device = "cpu"
        # Multi-modal wrappers carry the decoder settings in `text_config`.
        self.text_config = getattr(hf_config, "text_config", hf_config)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.text_config.dtype)
        torch.set_default_device(self.device)
        self.model = build_model(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # speculative (MTP) decoding state
        self.spec_decode = config.spec_decode and getattr(self.model, "mtp", None) is not None
        if config.spec_decode and not self.spec_decode:
            raise ValueError("spec_decode=True but the model has no MTP drafter")
        self.linear_modules: list[GatedDeltaNet] = [
            m for m in self.model.modules() if isinstance(m, GatedDeltaNet)
        ]
        # per-seq drafter inputs stashed between steps: (token_ids, positions, target_hiddens)
        self.spec_stash: dict[int, tuple[list[int], list[int], list[torch.Tensor]]] = {}
        # acceptance statistics: hist[a] = verify steps where exactly a of k drafts passed
        self.spec_stats = {"steps": 0, "hist": [0] * (config.spec_num_draft_tokens + 1)}
        # last target hidden per seq, chaining hidden states across chunked prefill
        self.spec_boundary: dict[int, torch.Tensor] = {}

        self.allocate_kv_cache()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # stable dense slot per sequence for linear-attention state buffering
        self.seq_to_slot: dict[int, int] = {}
        self.free_slots: list[int] = list(range(config.max_num_seqs))

    def exit(self):
        pass

    def loop(self):
        pass

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)

    def release_seq(self, seq_id: int):
        """Free per-seq runner resources once the sequence is finished."""
        self.spec_stash.pop(seq_id, None)
        self.spec_boundary.pop(seq_id, None)
        slot = self.seq_to_slot.pop(seq_id, None)
        if slot is not None:
            if self.linear_modules:
                self.linear_state[:, slot].zero_()
                self.linear_conv_state[:, slot].zero_()
            self.free_slots.append(slot)

    def reset_spec_stats(self):
        self.spec_stats = {"steps": 0, "hist": [0] * (self.config.spec_num_draft_tokens + 1)}

    def format_spec_stats(self) -> str:
        """Acceptance statistics since the last reset (reset_spec_stats / engine start)."""
        steps = self.spec_stats["steps"]
        hist = self.spec_stats["hist"]
        if not steps:
            return "spec stats: no verify steps recorded"
        k = len(hist) - 1
        # draft i is evaluated exactly in the steps where drafts 0..i-1 passed, i.e.
        # steps whose num_accepted >= i — all rates follow from the histogram alone
        evaluated = [sum(c for a, c in enumerate(hist) if a >= i) for i in range(k + 1)]
        drafts_eval = sum(evaluated[:k])
        drafts_acc = sum(a * c for a, c in enumerate(hist))
        emitted = sum((a + 1) * c for a, c in enumerate(hist))
        pos = "  ".join(
            f"d{i + 1}: {evaluated[i + 1] / evaluated[i]:5.1%} ({evaluated[i + 1]}/{evaluated[i]})"
            for i in range(k))
        hist_s = "  ".join(f"acc={a}: {c} ({c / steps:.1%})" for a, c in enumerate(hist) if c)
        return (f"verify steps: {steps} | drafts accepted: {drafts_acc}/{drafts_eval} "
                f"({drafts_acc / drafts_eval:.1%}) | emitted: {emitted} tokens "
                f"({emitted / steps:.2f}/verify step)\n"
                f"  per-position: {pos}\n"
                f"  histogram: {hist_s}")


    def allocate_kv_cache(self):
        config = self.config
        tc = self.text_config
        layer_types = getattr(tc, "layer_types", None)
        num_full_attn = layer_types.count("full_attention") if layer_types else tc.num_hidden_layers
        if self.spec_decode and getattr(self.model, "mtp", None) is not None:
            num_full_attn += len(self.model.mtp.layers)    # drafter has its own KV cache
        num_heads = tc.num_key_value_heads
        head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
        block_bytes = 2 * num_full_attn * self.block_size * num_heads * head_dim * tc.dtype.itemsize

        max_possible_blocks = config.max_num_seqs * ((config.max_model_len + self.block_size - 1) // self.block_size)
        import os
        cpu_kv_gb = float(os.environ.get("NANOVLLM_CPU_KV_GB", "4"))
        cpu_kv_budget = int(cpu_kv_gb * 1024 ** 3)
        config.num_kvcache_blocks = min(max_possible_blocks, max(1, cpu_kv_budget // block_bytes))
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, num_full_attn, config.num_kvcache_blocks, self.block_size, num_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                if layer_id >= num_full_attn:    # drafter attention when spec decode is off
                    continue
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        # Linear-attention (GatedDeltaNet) recurrent + conv state, keyed by sequence slot.
        if isinstance(self.model, Qwen3_5ForCausalLM):
            num_linear = sum(
                1 for m in self.model.modules() if isinstance(m, GatedDeltaNet) and m.recurrent_state.numel() == 0
            )
            if num_linear:
                max_seqs = config.max_num_seqs
                v_heads = tc.linear_num_value_heads
                k_dim = tc.linear_key_head_dim
                v_dim = tc.linear_value_head_dim
                k_heads = tc.linear_num_key_heads
                conv_dim = k_dim * k_heads * 2 + v_dim * v_heads
                state_len = tc.linear_conv_kernel_dim - 1
                self.linear_state = torch.zeros(num_linear, max_seqs, v_heads, k_dim, v_dim, dtype=torch.float32)
                self.linear_conv_state = torch.zeros(num_linear, max_seqs, conv_dim, state_len, dtype=tc.dtype)
                linear_id = 0
                for module in self.model.modules():
                    if isinstance(module, GatedDeltaNet):
                        module.recurrent_state = self.linear_state[linear_id]
                        module.conv_state = self.linear_conv_state[linear_id]
                        if self.spec_decode:
                            k = self.config.spec_num_draft_tokens
                            module.verify_snap_rec = torch.zeros(k, max_seqs, v_heads, k_dim, v_dim, dtype=torch.float32)
                            module.verify_snap_conv = torch.zeros(k, max_seqs, conv_dim, state_len, dtype=tc.dtype)
                        linear_id += 1

    def _tensor(self, data, dtype):
        t = torch.tensor(data, dtype=dtype)
        return t

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return self._tensor(block_tables, torch.int32)

    def _slot(self, seq: Sequence, fresh: bool = False) -> int:
        slot = self.seq_to_slot.get(seq.seq_id)
        if slot is None:
            slot = self.free_slots.pop()
            self.seq_to_slot[seq.seq_id] = slot
        if fresh:    # (re-)prefill from scratch: drop any stale linear-attention state
            if self.linear_modules:
                self.linear_state[:, slot].zero_()
                self.linear_conv_state[:, slot].zero_()
        return slot

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        seq_slots = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            slot = self._slot(seq, fresh=start == 0)
            seq_slots.extend([slot] * seqlen_q)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = self._tensor(input_ids, torch.int64)
        positions = self._tensor(positions, torch.int64)
        cu_seqlens_q = self._tensor(cu_seqlens_q, torch.int32)
        cu_seqlens_k = self._tensor(cu_seqlens_k, torch.int32)
        slot_mapping = self._tensor(slot_mapping, torch.int32)
        seq_slots = self._tensor(seq_slots, torch.int32)
        # kept so the MTP drafter prefill can reuse the exact same context
        self._prefill_ctx = (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, seq_slots)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        seq_slots = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
            seq_slots.append(self._slot(seq))
        input_ids = self._tensor(input_ids, torch.int64)
        positions = self._tensor(positions, torch.int64)
        slot_mapping = self._tensor(slot_mapping, torch.int32)
        context_lens = self._tensor(context_lens, torch.int32)
        seq_slots = self._tensor(seq_slots, torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, seq_slots=seq_slots)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        return self._tensor(temperatures, torch.float32)

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        if is_prefill:
            return self._run_prefill(seqs)
        if self.spec_decode:
            return self._run_decode_spec(seqs)
        return self._run_decode(seqs)

    def _run_prefill(self, seqs: list[Sequence]) -> list[list[int]]:
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs)
        hidden = self.model(input_ids, positions)             # [T, H], final-norm output
        logits = self.model.lm_head(hidden)                   # [N, V], last position per seq
        token_ids = self.sampler(logits, temperatures).tolist()
        if self.spec_decode:
            self._draft_prefill(seqs, input_ids, positions, hidden, token_ids)
        reset_context()
        return [[t] for t in token_ids]

    def _draft_prefill(self, seqs, input_ids, positions, hidden, token_ids):
        """Run the drafter over each scheduled chunk to fill its KV cache, and stash the
        drafter-extend inputs (draft of the next token) for sequences finishing prefill.

        The drafter pair at position j is (embed(t_j), h_{j-1}) — the MTP training
        convention — so the target hidden stream is shifted by one position.
        """
        cu_q, cu_k, max_q, max_k, slot_mapping, block_tables = self._prefill_ctx
        set_context(True, cu_q, cu_k, max_q, max_k, slot_mapping, None, block_tables, None)
        shifted_hiddens = []
        for i, seq in enumerate(seqs):
            s = cu_q[i].item()
            n = seq.num_scheduled_tokens
            e = s + n
            h_prev = self.spec_boundary.pop(seq.seq_id, None)    # h_{start-1} from previous chunk
            if h_prev is None:
                h_prev = hidden[s:s + 1]                         # boundary: approximate with h_start
            shifted_hiddens.append(h_prev)
            if n > 1:
                shifted_hiddens.append(hidden[s:e - 1])
            self.spec_boundary[seq.seq_id] = hidden[e - 1:e]
            if seq.num_cached_tokens + n == seq.num_tokens:      # final chunk -> token committed
                self.spec_stash[seq.seq_id] = (
                    [token_ids[i]], [seq.num_cached_tokens + n], [hidden[e - 1]],
                )
        target_hidden = torch.cat(shifted_hiddens, dim=0)
        self.model.mtp(input_ids, positions, target_hidden)

    def _run_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        input_ids, positions = self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.model.compute_logits(self.model(input_ids, positions))
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return [[t] for t in token_ids]

    def _verify_slot(self, seq: Sequence, pos: int) -> int:
        return seq.block_table[pos // self.block_size] * self.block_size + pos % self.block_size

    def _verify_fwd(self, seqs, seq_slots, tokens, pos_off, q_per_seq):
        """Varlen verify forward with q_per_seq tokens per sequence, starting at
        position len(seq)-1+pos_off. Returns per-token hiddens, token-major."""
        input_ids, positions, slot_mapping = [], [], []
        cu_q, cu_k = [0], [0]
        max_q = max_k = 0
        seq_slots_flat = []
        for i, seq in enumerate(seqs):
            p = len(seq) - 1 + pos_off
            toks = tokens[i * q_per_seq:(i + 1) * q_per_seq]
            input_ids.extend(toks)
            positions.extend(range(p, p + q_per_seq))
            slot_mapping.extend(self._verify_slot(seq, pos) for pos in range(p, p + q_per_seq))
            seq_slots_flat.extend([seq_slots[i]] * q_per_seq)
            cu_q.append(cu_q[-1] + q_per_seq)
            cu_k.append(cu_k[-1] + p + q_per_seq)
            max_q = max(max_q, q_per_seq)
            max_k = max(max_k, p + q_per_seq)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, self._tensor(cu_q, torch.int32), self._tensor(cu_k, torch.int32),
                    max_q, max_k, self._tensor(slot_mapping, torch.int32), None, block_tables,
                    self._tensor(seq_slots_flat, torch.int32), is_verify=True)
        return self.model(self._tensor(input_ids, torch.int64), self._tensor(positions, torch.int64))

    def _run_decode_spec(self, seqs: list[Sequence]) -> list[list[int]]:
        temperatures = self.prepare_sample(seqs)
        k = self.config.spec_num_draft_tokens
        n = len(seqs)

        # ---- 1) drafter: extend over the newly committed tokens (target hiddens
        #         lag one position), then autoregressively draft k tokens — the
        #         drafter consumes its own hidden at the speculative positions.
        ext_tokens, ext_positions, ext_hiddens = [], [], []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0
        slot_mapping = []
        for seq in seqs:
            tokens, positions, hiddens = self.spec_stash.pop(seq.seq_id)
            first_new = positions[0]
            count = len(tokens)
            ext_tokens.extend(tokens)
            ext_positions.extend(positions)
            ext_hiddens.extend(hiddens)
            cu_q.append(cu_q[-1] + count)
            cu_k.append(cu_k[-1] + first_new + count)
            max_q = max(max_q, count)
            max_k = max(max_k, first_new + count)
            slot_mapping.extend(self._verify_slot(seq, pos) for pos in positions)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, self._tensor(cu_q, torch.int32), self._tensor(cu_k, torch.int32),
                    max_q, max_k, self._tensor(slot_mapping, torch.int32), None, block_tables, None)
        mtp_out = self.model.mtp(
            self._tensor(ext_tokens, torch.int64),
            self._tensor(ext_positions, torch.int64),
            torch.stack(ext_hiddens, dim=0),
        )
        draft_l = self.model.lm_head(mtp_out)                 # [N, V], last position per seq
        draft_h = mtp_out[[c - 1 for c in cu_q[1:]]]          # drafter hidden at those positions
        draft_tokens = [self.sampler(draft_l, temperatures)]
        draft_logits = [draft_l]
        for i in range(1, k):
            positions_i = [len(seq) - 1 + i for seq in seqs]
            slot_i = [self._verify_slot(seq, p) for seq, p in zip(seqs, positions_i)]
            cu_k_i = [0]
            mx = 0
            for seq, p in zip(seqs, positions_i):
                cu_k_i.append(cu_k_i[-1] + p + 1)
                mx = max(mx, p + 1)
            set_context(True, self._tensor(list(range(n + 1)), torch.int32), self._tensor(cu_k_i, torch.int32),
                        1, mx, self._tensor(slot_i, torch.int32), None, block_tables, None)
            mtp_out = self.model.mtp(
                self._tensor([int(t) for t in draft_tokens[-1]], torch.int64),
                self._tensor(positions_i, torch.int64),
                draft_h,
            )
            draft_l = self.model.lm_head(mtp_out)             # [N, V]
            draft_h = mtp_out                                 # [N, H], one row per seq
            draft_tokens.append(self.sampler(draft_l, temperatures))
            draft_logits.append(draft_l)
        reset_context()

        # ---- 2) target verify:
        #         fwd1: last committed token @ p         -> state after p (snapshot 0)
        #         fwd2: the k drafts        @ p+1..p+k   -> state after p+k
        #       GatedDeltaNet snapshots its state after every draft position so any
        #       accepted prefix can be restored when the chain is cut short.
        seq_slots = [self._slot(seq) for seq in seqs]
        slots_t = self._tensor(seq_slots, torch.int64)
        hidden1 = self._verify_fwd(seqs, seq_slots, [seq.last_token for seq in seqs], 0, 1)
        saved = [(m.recurrent_state[slots_t].clone(), m.conv_state[slots_t].clone())
                 for m in self.linear_modules]                 # snapshot 0
        drafts = torch.stack(draft_tokens, dim=1)              # [N, K]
        hidden2 = self._verify_fwd(seqs, seq_slots, drafts.view(-1).tolist(), 1, k)
        reset_context()
        logits = torch.cat([self.model.lm_head(hidden1).unsqueeze(1),
                            self.model.lm_head(hidden2).view(n, k, -1)], dim=1)   # [N, K+1, V]

        # ---- 3) chain rejection sampling over the k drafts.
        num_acc, corrected, bonus = self.sampler.spec_verify(
            logits, torch.stack(draft_logits, dim=1), temperatures, drafts)
        na = num_acc.tolist()
        st = self.spec_stats
        st["steps"] += n
        for a in na:
            st["hist"][a] += 1
        corrected = corrected.tolist()
        bonus = bonus.tolist()
        drafts = drafts.tolist()

        # roll the linear state back to the last accepted draft position
        for i in range(n):
            j = na[i]                                          # drafts 0..j-1 accepted
            if j == k:
                continue
            for li, m in enumerate(self.linear_modules):
                if j == 0:
                    rec, conv = saved[li]
                    m.recurrent_state[seq_slots[i]] = rec[i]
                    m.conv_state[seq_slots[i]] = conv[i]
                else:
                    m.recurrent_state[seq_slots[i]].copy_(m.verify_snap_rec[j - 1, seq_slots[i]])
                    m.conv_state[seq_slots[i]].copy_(m.verify_snap_conv[j - 1, seq_slots[i]])

        # ---- 4) build per-seq outputs and stash the next drafter-extend inputs.
        eos = self.config.eos
        outputs = []
        for i, seq in enumerate(seqs):
            p = len(seq) - 1
            # hidden pool: h(t_p) then the k draft hiddens; hiddens[m] is the target
            # hidden of the token preceding toks[m]
            h_pool = [hidden1[i]] + [hidden2[i * k + m] for m in range(k)]
            j = na[i]
            toks = drafts[i][:j] + ([corrected[i]] if j < k else [bonus[i]])
            max_new = seq.max_tokens - seq.num_completion_tokens
            toks = toks[:max(1, max_new)]
            if not seq.ignore_eos:
                for jj, t in enumerate(toks):
                    if t == eos:
                        toks = toks[:jj + 1]
                        break
            outputs.append(toks)
            finished = bool(toks) and (
                (not seq.ignore_eos and toks[-1] == eos)
                or seq.num_completion_tokens + len(toks) >= seq.max_tokens
            )
            if finished:
                continue
            self.spec_stash[seq.seq_id] = (
                toks, list(range(p + 1, p + 1 + len(toks))), h_pool[:len(toks)])
        return outputs