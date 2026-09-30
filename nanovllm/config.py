import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    spec_decode: bool = False
    spec_num_draft_tokens: int = 1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # Multi-modal wrappers carry the decoder settings in `text_config`.
        text_cfg = getattr(self.hf_config, "text_config", self.hf_config)
        self.max_model_len = min(self.max_model_len, getattr(text_cfg, "max_position_embeddings", self.max_model_len))
        if self.spec_decode:
            num_mtp = getattr(text_cfg, "mtp_num_hidden_layers", 0)
            assert num_mtp == 1, f"spec_decode requires a checkpoint with exactly 1 MTP layer, got {num_mtp}"
            assert 1 <= self.spec_num_draft_tokens <= 8, \
                f"spec_num_draft_tokens must be in [1, 8], got {self.spec_num_draft_tokens}"
