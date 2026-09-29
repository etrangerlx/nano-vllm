import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

c = AutoConfig.from_pretrained("E:/models/Qwen3.5-0.8B")
tc = c.text_config
print("type", type(tc).__name__)
print("QCK", isinstance(tc, Qwen3_5TextConfig))
for a in [
    "hidden_size", "attention_bias", "rms_norm_eps", "layer_types",
    "linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim",
    "linear_value_head_dim", "linear_conv_kernel_dim", "intermediate_size",
    "rope_parameters", "num_attention_heads", "num_key_value_heads",
    "head_dim", "vocab_size", "num_hidden_layers", "tie_word_embeddings",
    "dtype", "max_position_embeddings", "hidden_act",
]:
    v = getattr(tc, a, "<MISSING>")
    print(a, "=", str(v)[:60])
# also check top-level combined config fields used elsewhere
print("top model_type", c.model_type)
print("top has max_position_embeddings", hasattr(c, "max_position_embeddings"))
print("top has dtype", hasattr(c, "dtype"))