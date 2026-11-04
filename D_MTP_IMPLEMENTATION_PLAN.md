# D-MTP 实现方案与计划

> 目标:在 speculators 中实现 **D-MTP** 的训练——冻结的预训练 MTP 模块提出块内首 token(T1),
> DFlash 块并行骨干续写其余候选,中间以"token 嵌入 × MTP 隐状态"融合模块衔接。
>
> 依据:技术博客 https://sylvan820.github.io/D-MTP/ (Shen 2026;论文与官方代码标注
> "to follow soon",**暂无公开检查点**,因此本实现无权重键对齐约束,命名遵循本仓库惯例)。

---

## 1. 方法定义(博客精要)

每轮草稿:冻结 MTP 先串行提出 T1 → DFlash 以含 T1 的块一次并行前向提出 T2..T7 →
目标模型一次验证 7 个候选(验证窗口 B=8,含 bonus)。

**融合模块**(唯一新增可训练组件,除 DFlash 本体外):

```
ẽ_T1 = W_f [ RMS_e(e_T1) ; RMS_m(h_MTP) ]      # 两个独立 RMSNorm + Linear(2d→d)
```

- `e_T1`:T1 的词嵌入;`h_MTP`:**预测出 T1 的那个** MTP 隐状态(非消费 T1 之后的)。
- `ẽ_T1` 仅替换块内 T1 槽位的输入嵌入;MASK 嵌入、DFlash 输出头、目标上下文 KV
  注入路径全部不变;MTP 的 KV 不并入 DFlash 上下文。

**训练**:目标模型与 MTP 均冻结,仅更新 DFlash 层与融合模块;T1 槽位为已知输入、
不计损失;6 个续写位置用 DFlash 原版位置衰减 CE。训练期 T1 取数据集真值
(teacher forcing),与之融合的是"预测该 T1"的 MTP 隐状态——时序语义与推理一致。

**块布局**:博客主用单锚 `[T1, MASK×6]`;双锚变体 `[A, T1, MASK×6]` 接受长度等价
(3.951 vs 3.963)。**本实现采用双锚变体**——它与本仓库现有 anchor 机制零冲突
(见 §3.2)。

**收益**:首 token 接受率 +5.4~16.6 pp;27B 公开复现接受长度较 DFlash +5.8%~21.6%;
每轮开销 +0.4~0.55 ms(一个 MTP 解码块 + LM head)。

## 2. 仓库现状与复用面

| 需求 | 仓库已有 | 复用方式 |
|---|---|---|
| 冻结 MTP 模块 | `models/mtp/model_definitions.py`:`mtp_model_classes`(qwen3 / qwen3_next / qwen3_5 / qwen3_5_moe)+ `convert/mtp/converter.py` 从 verifier 提取 `mtp.*` 权重 | 直接实例化 `first_layer_class` + 同族 rotary;Converter 装权重后 `requires_grad_(False)` |
| h_MTP 所需输入 | 训练 batch 已含 `verifier_last_hidden_states` 与 `input_ids` | **训练期在线计算,数据管线零改动**(MTP 仅一层,且 no_grad) |
| 块并行骨干 | `DFlashDraftModel`(anchor 采样、块注意力 mask、冻结 verifier head、D-PACE/CE 损失族) | 继承;dcore 已铺好的钩子模式复用 |
| 训练接线 | dcore 刚走通的 registry/schema/resolution 模式 | 照搬 |

**MTP 语义核对**(`models/mtp/core.py` forward):`MTP(h_t, emb(x_{t+1})) → 预测 x_{t+2}`。
故锚点 p(已提交 token)处,T1 = x_{p+1},**h_MTP = 序列索引 t = p−1 处的 MTP 输出**
(由 h_{p−1} 与 emb(x_p) 算得)。推理时 x_p 刚提交、h_{p−1} 来自上一验证步——一致。

## 3. 设计

### 3.1 形态:新增 `dmtp` speculator 类型

```
src/speculators/models/dmtp/
├── __init__.py
├── config.py              # DMTPSpeculatorConfig(DFlashSpeculatorConfig)
├── core.py                # DMTPDraftModel(DFlashDraftModel)
└── model_definitions.py   # MTPTokenFusion
```

### 3.2 块布局与损失(双锚变体,block_size=8)

现有 DFlash 机制:块锚定于 p,槽位 p..p+B−1,targets 为"槽位 k 预测 x_{p+k} 的分布",
`sample_from_anchor=False` 时槽 0(锚)不计损失,`speculative_tokens = B−1`。

D-MTP 双锚布局**完全落在该机制内**:

| 槽位 | 内容 | 损失 |
|---|---|---|
| 0 | 锚 A = x_p(已提交) | 无(现有逻辑) |
| 1 | T1 = x_{p+1},嵌入替换为 `ẽ_T1` | **无(新增掩零)** |
| 2..7 | MASK,预测 x_{p+2}..x_{p+7} | 位置衰减 CE(现有) |

targets、注意力 mask、anchor 采样、`speculative_tokens=7` 全部不变;
唯二改动 = 槽 1 的输入嵌入 + 槽 1 的损失掩零。

### 3.3 模块与配置

**`MTPTokenFusion`**(model_definitions.py):`emb_norm: RMSNorm(d)`、
`mtp_norm: RMSNorm(d)`、`fusion_proj: Linear(2d→d, bias=False)`。

**`DMTPSpeculatorConfig`**:继承 DFlash 全部字段,新增
`mtp_transformer_config: PretrainedConfig`——构建冻结 MTP 块所需的 **verifier 架构
config**(draft 是 5 层 qwen3 小解码器,而 MTP 块必须按 verifier 本体架构建,
如 qwen3_5;二者只共享 hidden_size 与词表)。该字段随 checkpoint 序列化,
使 `from_pretrained` 无需访问 verifier 即可重建模块树。

**`DMTPDraftModel`** 关键点:

1. `__init__`:构建 `self.fusion`、`self.mtp_layers = ModuleList([first_layer_class(mtp_tc, 0)])`、
   `self.mtp_rotary_emb`;MTP 参数 `requires_grad_(False)`。
   **MTP 权重随 checkpoint 保存**(沿 `MTPDraftModel` 惯例——其 `_keys_to_ignore_on_save`
   仅排除 embed/lm_head),保证 `from_pretrained` 自洽。
2. `from_training_args`:`_build_base_config_kwargs("dmtp", ...)` + 用
   `get_verifier_config(verifier_name_or_path)` 填 `mtp_transformer_config`;
   建模后经 `MTPConverter().convert_to_state_dict()` 装入 `mtp.*` 权重再冻结。
3. **h_MTP 在线计算**(no_grad):对整段打包序列跑一次 MTP 块——
   `mtp_out = MTP(hidden=verifier_last[:, :L−1], token_embeddings=emb(input_ids[:, 1:]),
   causal mask, rotary(position_ids))`;`mtp_out[:, t]` 预测 x_{t+2};
   锚点 p 的 `h_MTP = mtp_out[:, (p−1).clamp(min=0)]`。
   Qwen3.5 MRoPE 经 `_prepare_mtp_position_ids` 复用处理。
4. **嵌入替换钩子**:在 `DFlashDraftModel._backbone_forward` 把
   `noise_embedding = self.embed_tokens(mask_token_ids)` 抽成
   `self._build_noise_embedding(mask_token_ids, input_ids, anchor_positions,
   verifier_last_hidden_states, position_ids)`(基类实现 = 原两行语义,
   dflash/dcore/dspark/xpress 行为不变)。DMTP 覆写:槽 1 填 T1 token id、
   整体 embed、算 h_MTP、`ẽ_T1 = fusion(e_T1, h_MTP)` 替换槽 1。
5. `forward`:调 `_backbone_forward` 后将 `aligned_loss_mask[:, 1::block_size] = 0`,
   再走现有 `compute_metrics`(CE + fixed-exp-decay;与 D-PACE 正交,可实验切换)。

### 3.4 训练接线与默认值

| 文件 | 改动 |
|---|---|
| `models/__init__.py` | 导出 DMTP 类 |
| `train/config/schema.py` | `_DEFAULT_LOSS_FN["dmtp"]="ce"`;`is_dflash_family` 加 `dmtp`(5 层);block_size 走通用默认 8;per-position 保持 `fixed-exp-decay`(博客配方,不进 `uses_dpace_default`) |
| `train/config/resolution.py` | `_ALGORITHM_GROUP_USERS["dflash"]` 加 `dmtp` |
| 其余 | registry 装饰器自动接入 train CLI |

**前置条件**:verifier 须含原生 `mtp.*` 权重(Qwen3-Next / Qwen3.5 系;
repo 已有 `examples/train/mtp_qwen3_5_9b_gsm8k_online.sh` 同类目标)。
无原生 MTP 的目标(如 Qwen3-8B)需先用 `--speculator-type mtp` 训出 MTP 头,
本期不展开。

### 3.5 已知限制

1. **跨文档注意力**:打包序列上 MTP 的 causal mask 不隔离文档——与仓库现有 MTP
   训练路径同一约定;position_ids 按文档重置可部分缓解。后续可改 document 感知 mask。
2. **推理侧**:需 vLLM/SGLang 实现"MTP 先行一步 + DFlash 续块"调度与融合模块,
   改动大于 dcore,独立工作流;本仓库内验证训练期 EAL/逐位置 acc(重点看位置 2 的
   acc,对应推理首个 DFlash 候选)。
3. **开销**:每训练步多一次单层 MTP 前向(no_grad,8K 序列),实测应 <5% 步时。

## 4. 实施清单与验收

| # | 项 | 验收 |
|---|---|---|
| 1 | dflash `_build_noise_embedding` 钩子化 | dflash 族既有单测零回归 |
| 2 | `models/dmtp/` 三件 + registry | 注册/构建/保存-加载往返单测 |
| 3 | 接线(schema/resolution/`__init__`) | `TrainConfig(speculator_type="dmtp")` 默认值单测;组表单测 |
| 4 | 单测 `test_dmtp.py`:融合模块形状与梯度;槽 1 嵌入确被替换(与纯 embed 不同、与手算 fusion 相等);h_MTP 索引正确性(与直接调 MTP 层逐位对照);槽 0/1 损失掩零;MTP 参数冻结;CPU 全前向+反向冒烟(qwen3 tiny 配置) | 全绿 |
| 5 | 示例脚本 `examples/train/dmtp_qwen3_5_9b_gsm8k_online.sh` + 文档 `docs/user_guide/algorithms/dmtp.md` + index 登记 | lint 通过 |

预估:2–3 天;本机(无 GPU)验证手段同 dcore:pytest(CPU)+ ruff。

## 5. 参考

- D-MTP 博客:https://sylvan820.github.io/D-MTP/
- DFlash:arXiv:2602.06036(github.com/z-lab/dflash)
- 本仓库 MTP 支持:`src/speculators/models/mtp/`,FastMTP 步权重见 arXiv:2509.18362
