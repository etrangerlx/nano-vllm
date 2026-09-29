# 实现 Qwen3.5-0.8B 纯文本推理

## Context（背景）

用户希望按现有 nano-vllm 项目为 `E:/models/Qwen3.5-0.8B` 实现推理。
该模型是图文多模态，经确认本次**只做纯文本推理**（后续再加 MTP）。

关键事实（已核实）：
- 模型 `model_type="qwen3_5"`，架构由 `text_config` 描述（`model_type="qwen3_5_text"`）。
- 文本 backbone 是**混合架构**，24 层：`layer_types` 中索引 `{3,7,11,15,19,23}` 为 `full_attention`（6 层），其余为 `linear_attention`（18 层）。
- `linear_attention` 是 **GatedDeltaNet（门控线性注意力）**，无 RoPE，靠卷积(conv1d)+递归状态。
- `full_attention` 是标准注意力，但带 `q_norm/k_norm`、RoPE 只旋转前 64 维（`head_dim=256 * partial_rotary_factor=0.25`），且输出带 `sigmoid(gate)` 门控。
- RMSNorm 公式与 Llama 不同：`xxx = rms(x) * (1 + weight)`，weight 初始为 0；`lm_head` 与 `embed_tokens` 权重绑定（tie）。
- 权重命名：文本权重带前缀 `model.language_model.`；视觉权 `model.visual.*`、MTP 权 `mtp.*` **不加载**（逻辑上跳过）。
- 参考实现：`D:\miniconda3\envs\env_LLM\lib\site-packages\transformers\models\qwen3_5\modeling_qwen3_5.py`（torch 2.10+cpu / transformers 5.16.1）。

目标：复用现有 nano-vllm 的批量/调度/采样框架，在不破坏 qwen2/qwen3 的前提下，让 `LLM(path)` 能对 Qwen3.5 做文本自回归生成，并与 transformers 参考逐 token 对照。

## 需要新增/修改的文件

### 1. 新增 `nanovllm/models/qwen3_5.py`
全部文本结构，参数名与权重名严格对齐（顶层 `embed_tokens / layers / norm / lm_head`，不加 `model` 封装）：

- `Qwen3_5RMSNorm(dim, eps)`：`weight=nn.Parameter(torch.zeros(dim))`；前向 `rms(x)*(1+weight)`，fp32 计算。
- `Qwen3_5RMSNormGated(dim, eps)`：`weight=ones`；`rmsnorm(x)*weight*silu(gate)`。
- `Qwen3_5PartialRotary`：`inv_freq = 1/base**(arange(0=64,2)/64)`，`base=10_000_000`（text_config.rope_parameters.rope_theta）；按 positions 即时算 `cos/sin`（shape `[N,64]`）。对 q/k 前 64 维做 `rotate_half` 旋转，其余 pass-through（对应参考 `apply_rotary_pos_emb`）。
- `GatedDeltaNet(config, layer_idx)`（线性注意力层）：
  - 参数：`conv1d(Conv1d, in/out=conv_dim, 无bias, kernel=4, groups)`，`in_proj_qkv / in_proj_z / in_proj_b / in_proj_a(QLinear 无bias)`，`norm(Qwen3_5RMSNormGated, head_v_dim)`，`out_proj(Linear value_dim→hidden 无bias)`，`A_log / dt_bias`（形状 `[num_v_heads]`）。
  - `conv_dim = key_dim*2+value_dim`，`key_dim=head_k_dim*num_k_heads`、`value_dim=head_v_dim*num_v_heads`；本模型 `head_k_dim=head_v_dim=128`、`num_k_heads=num_v_heads=16`。
  - 前向（is_prefill / decode 分支）：
    - `mixed_qkv=conv1d(in_proj_qkv(x).transpose(1,2))`（用 `F.conv1d`，权重 `[conv_dim,1,4]`）。
    - 拆 `q,k,v`；`z=in_proj_z(x)`，`b=in_proj_b(x).sigmoid()`，`g=-exp(A_log)*softplus(in_proj_a(x)+dt_bias)`。
    - reshape 到 head 维度；`q=repeat_interleave(num_v_heads//num_k_heads)`。
    - **prefill**：按 `context.cu_seqlens_q` scatter 到 `[B,max_seqlen,D]`，用**分块版** chunk_gated_delta_rule（复刻 `torch_chunk_gated_delta_rule`），初始递归态取自 `状态槽 buffer[slot]`，结束后写回 `buffer[slot]`；再 gather 回 flat。
    - **decode**：每 seq 1 token，用**递归式** torch_recurrent_gated_delta_rule 单步，初始态取 `buffer[slot]`，写回。
    - `out = out_proj(norm(core_attn_out, z))`。
  - 用 l2norm 归一化 q/k（`use_qk_l2norm_in_kernel=True`）。
- `Qwen3_5Attention(config, layer_idx)`（full attention 层，复用现有 `nanovllm.layers.attention.Attention` 做 KV 缓存与 SDPA）：
  - `q_proj(hidden→ num_heads*head_dim*2)`，`k_proj/v_proj(all→ num_kv_heads*head_dim)`，`o_proj(num_heads*head_dim→hidden)`，`q_norm/k_norm(Qwen3_5RMSNorm, head_dim)`。
  - `q,gate=chunk(q_proj(x).view(-1,nh,hd*2),2)`→ gate reshape `[-1,nh*hd]`；`q_norm(q)`、`k_norm(k)`、`v`；`PartialRotary` 旋 q/k；调 `Attention(q,k,v)`；`out*out.sigmoid(gate)`；`o_proj`。
- `Qwen3_5MLP`：`gate_proj/up_proj(hidden→intermediate=3584)`、`down_proj(intermediate→hidden)` 均无 bias；`down(silu(gate)*up)`。
- `Qwen3_5DecoderLayer`：按 `config.layer_types[layer_idx]` 创建 `linear_attn` 或 `self_attn`；`input_layernorm / post_attention_layernorm(Qwen3_5RMSNorm)`；**非融合残差**：`h=input_layernorm(x)`→mixer→`x=residual+h`；`h=post_attention_layernorm(x)`→mlp→`x=residual+h`。
- `Qwen3_5ForCausalLM(Qwen3_5Config)`：
  - `self.embed_tokens=VocabParallelEmbedding(vocab,hidden)`、`self.layers`、`self.norm(Qwen3_5RMSNorm)`、`self.lm_head=ParallelLMHead`，tie：`lm_head.weight.data=embed_tokens.weight.data`。
  - `forward(input_ids, positions)`→embeds→层循环→`norm`。
  - `compute_logits(hidden)`→`lm_head`。
  - 实例属性 `weight_prefix = "model.language_model."`。

### 2. 修改 `nanovllm/utils/loader.py`
- 在遍历权重名时：先 `strip model.weight_prefix`（存在则去前缀），再匹配/`get_parameter`。
- 找不到对应参数时**跳过**该权重（视觉 `model.visual.*`、MTP `mtp.*` 会被自然跳过）；qwen2/qwen3 逻辑保持不变（无前缀、全匹配）。

### 3. 修改 `nanovllm/config.py`
- 加一个取文本配置的辅助：`tc = getattr(hf_config, "text_config", hf_config)`。
- `__post_init__` 的 `max_position_embeddings` 改用 `tc`（Qwen3.5 顶层无该字段；仍对 qwen2/3 等兼容）。

### 4. 修改 `nanovllm/utils/context.py`
- 给 `Context` 增加 `seq_slots: torch.Tensor | None = None` 字段，`set_context` 增加一个形参，`reset_context` 归零。

### 5. 核心修改 `nanovllm/engine/model_runner.py`
- 引入 `nanovllm.models.qwen3_5.Qwen3_5ForCausalLM`；`build_model` 中 `model_type=="qwen3_5"` 时用 `getattr(hf_config,"text_config",None)` 构造。
- 统一取 `tc = getattr(hf_config,"text_config",hf_config)`：
  - 默认 dtype 用 `tc.dtype`；`allocate_kv_cache` 的 `num_kv_heads/head_dim/num_hidden_layers/dtype.itemsize` 用 `tc`。
  - `num_full_attn` = 有 `layer_types` 时统计 `full_attention` 数量，否则 `num_hidden_layers`；kv_cache 第 2 维与 `block_bytes` 用它。
- **线性注意力状态槽**：
  - 维护 `self.seq_to_slot: dict[int,int]`（seq_id→密集槽）与空闲槽集合。
  - 分配两个 buffer：`linear_state`（递归态，`[num_linear_layers, max_num_seqs, num_v_heads, head_k_dim, head_v_dim]`，fp32）与 `linear_conv_state`（`[num_linear_layers, max_num_seqs, conv_dim, kernel-1]`，注意 conv 态用参考的 state_len=kernel-1 约定）。
  - `run(seqs, is_prefill)` 时给无槽 seq 分配槽，构建与 token/seq 对齐的 `seq_slots` tensor 传入 `set_context(...)`；线性层据此读写对应槽状态。序列结束不急切回收（槽数 ≤ max_num_seqs，示例为少量请求）。

## 验证方式
1. 准备环境：用 `D:\miniconda3\envs\env_LLM\python.exe`（torch 2.10+cpu, transformers 5.16.1）。
2. 单元对照脚本（临时，不落库可删）：用 transformers 的 `Qwen3_5ForConditionalGeneration` 对纯文本 `input_ids` 取 logits，与 `nanovllm` 的 `Qwen3_5ForCausalLM.compute_logits` 对比：
   - 单个短文本、无权重随机数对齐困难 → 用**真实加载权重**后比较 logits 误差（只做 prefill，单 batch 与多 batch）。
   - decode 逐步对照（cache 正确性：递归态随步累积）。
3. 跑通 `example.py`（其指向 `E:/models/Qwen3.5-0.8B`），观察生成文本合理性；必要时调小 `max_num_seqs` 控制内存。
4. 确认 qwen2/qwen3 未回归（不跑权重即保持 import 不报错、逻辑分支不变）。

## 备注
- MTP 本次不做（transformers 运行时也不使用 MTP，仅预加载忽略）。
- 视觉 `model.visual.*` 与 `mtp.*` 权重在 loader 中被跳过。
- 纯 CPU + bf16 推理可能较慢，属预期；量级正确性优先。