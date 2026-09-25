import pickle
import torch
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen2 import Qwen2ForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def build_model(hf_config):
    if hf_config.model_type == "qwen2":
        return Qwen2ForCausalLM(hf_config)
    if hf_config.model_type == "qwen3":
        return Qwen3ForCausalLM(hf_config)
    raise ValueError(f"Unsupported model_type: {hf_config.model_type}")


class ModelRunner:

    def __init__(self, config: Config, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.event = event

        self.device = "cpu"

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(self.device)
        self.model = build_model(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.allocate_kv_cache()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)


    def exit(self):
        pass

    def loop(self):
        pass

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)


    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        
        max_possible_blocks = config.max_num_seqs * ((config.max_model_len + self.block_size - 1) // self.block_size)
        import os
        cpu_kv_gb = float(os.environ.get("NANOVLLM_CPU_KV_GB", "4"))
        cpu_kv_budget = int(cpu_kv_gb * 1024 ** 3)
        config.num_kvcache_blocks = min(max_possible_blocks, max(1, cpu_kv_budget // block_bytes))
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _tensor(self, data, dtype):
        t = torch.tensor(data, dtype=dtype)
        return t

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return self._tensor(block_tables, torch.int32)

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
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
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = self._tensor(input_ids, torch.int64)
        positions = self._tensor(positions, torch.int64)
        slot_mapping = self._tensor(slot_mapping, torch.int32)
        context_lens = self._tensor(context_lens, torch.int32)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        return self._tensor(temperatures, torch.float32)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
            
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids

    
