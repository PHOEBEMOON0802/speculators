# DCore 实现方案与计划

> 目标：在 speculators 项目中实现 **DCore** —— DFlash 高效块并行骨干 + DFlare 目标层上下文混合（Adaptive Layer Fusion + 异构 KV 投影）+ D-PACE 训练目标。
>
> 依据论文：
> - 《When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding》(arXiv:2610.24993v3，DCore 出处)
> - 《DFLARE: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding》(arXiv:2606.02091v2，目标层上下文混合出处)
> - 《D-PACE: Dynamic Position-Aware Cross-Entropy for Parallel Speculative Drafting》(arXiv:2605.18810，训练目标出处)

---

## 1. DCore 方法定义（论文精要）

DCore（2610.24993v3 附录 G）是三个组件的组合，**不新增任何推理算子**（相对 DFlare）：

| 组件 | 来源 | 内容 |
|---|---|---|
| 骨干 | DFlash | 5 层块并行（block-diffusion）draft 解码器；anchor + mask-token 块输入；目标隐状态经 KV 注入；context-only anchor（anchor 为 bonus token，不预测）；冻结目标 LM head |
| 上下文混合 | DFlare | 每个 draft 层对 **9 个目标层**的隐状态学习自己的 softmax 标量加权和（替代 DFlash 的共享 FC 融合），并使用**异构 KV 投影**（目标上下文与 draft 块 token 使用独立的 K/V 投影矩阵） |
| 训练目标 | D-PACE | 位置自适应加权交叉熵：权重为"draft 存活到该位置的概率"的可微代理（detach），替代固定指数衰减 |
| 每预算一个检查点 | DCore | 每个验证预算 B（block_size）单独训练一个 checkpoint |

论文结论：在 Qwen3-8B、B=16、Tt|Td=1|0 下，DCore 相对 DFlash 将平均加速比从 3.011× 提升到 3.242×，草稿延迟不变（1.79 ms）。D-PACE 对 CE 的增益集中在低熵任务（+7.9% vs +3.3%）。

### 1.1 DFlare 目标层混合的精确公式（2606.02091v2 §4）

**Adaptive Layer Fusion**（式 3）：设 `h_t^(j) ∈ R^d` 为第 j 个选定目标层（j=1..T，T=9）在位置 t 的隐状态，可学习矩阵 `W_fuse ∈ R^{D×T}`（D 为 draft 层数）：

```
α^(i) = softmax(W_fuse[i, :]) ∈ R^T
f_t^(i) = RMSNorm( Σ_j α_j^(i) · h_t^(j) )          # 每个 draft 层 i 一个专属融合视图
```

- 仅 D×T 个标量参数（5×9=45 个）；推理时 α 可预计算缓存，退化为标量加权和，延迟几乎为零。
- 消融（DFlare 表 2）：去掉 softmax 归一化退化最明显；融合后再加 FC 无增益（故 **DCore 不保留 DFlash 的共享 fc**）。

**异构 KV 投影**（式 4）：目标上下文与块内 token 使用独立投影：

```
K = [ F^(i)·Wt_K ;  x_t·Wd_K ;  X̂·Wd_K ]
V = [ F^(i)·Wt_V ;  x_t·Wd_V ;  X̂·Wd_V ]
Q = [x_t; X̂]·W_Q
```

- `Wt_K, Wt_V ∈ R^{d×d_kv}`：每个 draft 层专属，服务目标上下文 F^(i)。
- `Wd_K, Wd_V`：即现有的 k_proj/v_proj，服务 anchor token 与 mask 位置。
- 参数量核对（DCore 论文表 4）：Qwen3-8B（d=4096, d_kv=1024）下 FUSION 列 41.9M = 5 层 × 2 × 4096×1024，即全部来自每层的 Wt_K/Wt_V；DFlash 的共享 fc 为 83.9M（5·d×d）。**DCore 的融合参数量反而更小。**

注意：DFlare 论文还有"渐进式位置加权损失"（γ 线性升温），但 **DCore 不采用它**——其位置加权被 D-PACE 完全替代（DCore 论文表 3）。

### 1.2 D-PACE 目标的精确公式（arXiv:2605.18810）

设 `q_i` 为 draft 在位置 i 对目标选定 token 的概率（CE 的 exp 负值），B 为块内可预测位置数：

```
q̃_i = (1−α)·q_i + α                 # 平滑，α=0.5（论文默认，α∈[0.3,0.7] 稳定）
P_j  = Π_{i≤j} q̃_i                   # 块内前缀存活概率
w̄_j = sg( Σ_{m=j}^{B} P_m )          # 后缀和，stop-gradient（detach）
L    = Σ_j w̄_j · (−log q_j)          # 加权交叉熵；权重不归一化
```

标签为目标模型（重新生成语料上）选定的 token；权重计算纯 draft 侧，无额外 target 前向。

---

## 2. 代码库现状盘点（Gap Analysis）

调研结论：**DFlash 骨干与 D-PACE 目标均已在本仓库实现并且已是 dflash 的默认配置**。DCore 的缺口集中在 DFlare 式目标层混合与推理侧支持。

### 2.1 已具备（无需开发）

| 组件 | 位置 | 状态 |
|---|---|---|
| DFlash 块并行骨干 | `src/speculators/models/dflash/core.py`（`DFlashDraftModel`，anchor 块 flex-attention、mask-token 块输入、冻结 verifier LM head、`sample_from_anchor=False` 即 context-only anchor） | ✅ 与论文一致 |
| KV 注入注意力 | `src/speculators/models/dflash/model_definitions.py`（`Qwen3DFlashAttention`：`k_ctx/v_ctx` 由 target_hidden 经**共享** k_proj/v_proj 投影后拼接） | ✅（DFlash 版；DCore 需改造） |
| D-PACE 位置加权 | `src/speculators/losses/utils.py::dpace_loss_decay`（平滑→前缀累乘→后缀 cumsum→detach，`torch.no_grad`）；经 `per_position_loss_weight="dpace"` 启用 | ✅ 与论文公式逐项一致 |
| D-PACE 默认配方 | `src/speculators/train/config/schema.py`：dflash 默认 `per_position_loss_weight="dpace"`、`loss_fn="ce"`、`block_size=16`；校验器强制 dpace 必配 ce、α∈(0,1]（默认 0.5） | ✅（RFC #979 最佳实践已落地） |
| 损失族 | `src/speculators/losses/`（ce/kl/rkl/jsd/tv/nla/lk_hybrid，fused Triton + eager 双实现，`compound_loss` 支持 JSON 加权组合） | ✅（含论文消融用的 KL+TV） |
| 训练管线 | `src/speculators/train/`（torchrun 入口、在线 vLLM 隐状态源 `hidden_states_source.py`、异步 dataloader、FSDP、muon/adamw） | ✅ |
| 隐状态采集 | 在线：vLLM `--target-layer-ids`（须与 vLLM `eagle_aux_hidden_state_layer_ids` 一致）；离线：`data_generation/offline.py` | ✅（当前默认 3 层，DCore 需 9 层） |
| 训练内评估 | `models/dflash/metrics.py`：EAL（期望接受长度）、逐位置 acc；`tests/e2e/regression/test_training_only_acceptance.py` | ✅ |
| 检查点转换 | `src/speculators/convert/dflash/converter.py`（外部 DFlash → speculators 格式） | ✅（DFlash 版） |
| 参考实现模式 | `dflash2`、`dspark`、`xpress` 均为继承 `DFlashDraftModel` 的变体，展示了"新算法 = 新子包 + 注册 + 接线"的完整模式 | ✅ 可照搬 |

### 2.2 缺失（需开发）

| # | 缺口 | 说明 |
|---|---|---|
| G1 | **Adaptive Layer Fusion** | 现为共享 `fc: Linear(T·d → d)` + 单个 `hidden_norm`，所有 draft 层吃同一个融合向量（`core.py:418-420,455-466`）。需改为每 draft 层专属 softmax 标量加权 + RMSNorm |
| G2 | **异构 KV 投影** | 现在 `k_ctx = self.k_proj(target_hidden)` 与块 token 共享投影（`model_definitions.py:116-119`）。需为目标上下文增设每层 `kt_proj/vt_proj` |
| G3 | **9 目标层配方** | 默认 `resolve_target_layer_ids` 给 3 层（`[2, n/2, n-3]`，EAGLE3 兼容）；DCore 需在 [2, n−3] 均匀取 9 层，且数据管线（vLLM 采集、磁盘、带宽）按 3 倍压力评估 |
| G4 | **dcore 模型类型注册** | 新 config/model 类 + registry + 训练 CLI/schema/resolution 接线 |
| G5 | **推理侧（vLLM）支持** | vLLM 现有 DFlash 推理实现不含 per-layer fusion / 异构 KV。端到端加速评估依赖 vLLM 侧新增 DCore 支持（独立工作流，见 §5 Phase 4） |
| G6 | **转换器与导出** | DCore 检查点的保存/加载约定（新权重键）、外部 checkpoint（如 AngelSlim DFlare）转换 |
| G7 | 测试与文档 | 单测/集成/回归/示例脚本/算法文档 |

---

## 3. 总体设计

### 3.1 形态：新增 `dcore` speculator 类型

仿照 `dflash2` 的组织方式，新增子包 `src/speculators/models/dcore/`，模型类继承 `DFlashDraftModel`，最大化复用 anchor 选择、attention mask 构建、verifier 权重加载、D-PACE 损失与 metrics：

```
src/speculators/models/dcore/
├── __init__.py
├── config.py              # DCoreSpeculatorConfig(DFlashSpeculatorConfig)
├── core.py                # DCoreDraftModel(DFlashDraftModel)
└── model_definitions.py   # AdaptiveLayerFusion, Qwen3DCoreAttention, Qwen3DCoreDecoderLayer
```

不修改 `DFlashDraftModel` 的行为（已有 dflash/dflash2/dspark/xpress 四个下游继承者，回归风险高）；只在必要处把 `_backbone_forward` 中"融合+逐层调用"小步重构出可覆写的钩子（见 §4.3）。

### 3.2 关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| 融合位置 | 每步 forward 先算 `f^(i)`（i=1..D），训练中 α 参与梯度；推理/导出时可预计算 α | 与 DFlare 一致；训练开销可忽略（T·d 次乘加/位置/层） |
| RMSNorm 归属 | **每 draft 层一个** `fusion_norm_i`（D×d 参数） | 论文式 3 的 RMSNorm 在 per-layer f^(i) 上；按层独立自由度更高，参数可忽略。实现时与 AngelSlim（github.com/Tencent/AngelSlim）开源实现核对一次 |
| 共享 fc | **删除**（dcore 不创建 fc/hidden_norm） | DFlare 消融：fusion 后加 FC 无增益；DCore 参数表无 FC |
| k_norm | 目标上下文 keys 与块 keys 复用同一 `k_norm`（Qwen3 per-head RMSNorm），与现有代码对 concat 后整体 norm 的行为一致 | 保持与 DFlash 实现连续；与 AngelSlim 核对 |
| RoPE | 不变：现有代码对 [ctx; block] 拼接后的 K 整体施加 RoPE，Q 只对块位置 | 行为与 DFlash/DFlare 推理一致 |
| anchor 策略 | `sample_from_anchor=False`（context-only anchor，dflash 默认） | DCore 论文附录 G 明确保留 context-only anchor（anchor-pred 消融 3.251× < 3.261×） |
| 词表 | 沿用现有 `draft_vocab_size` 机制，**默认全词表**（论文用冻结目标 LM head），同时保留缩减词表选项 | 论文配方 = 冻结目标 head；repo 已支持两种，全词表时 lm_head 不保存 |
| 损失 | 复用 dflash 组：默认 `loss_fn="ce"` + `per_position_loss_weight="dpace"` + `dpace_alpha=0.5` | 已实现、已验证、已是默认 |
| 预算 | `block_size` 默认 16；按论文"每个验证预算单独训练 checkpoint" | 与现有 dflash 默认一致 |

---

## 4. 详细代码改动

> **⚠ 对齐修订（2026-11-01）**：对照官方开源检查点 `ernie-research/DCore-Qwen3-8B-Budget8`
> （本地 `C:\Users\Omni\cy_ws\DCore-Qwen3-8B-Budget8`，含参考实现 `dflare.py`），
> 为保证训练出的权重与官方检查点**键名/形状/初始化完全对齐**，对 §4.1–4.3 做如下修订
> （后文原始设计中与此冲突之处，以本节为准）：
>
> 1. **上下文 KV 投影命名**：`layers.{i}.self_attn.k_proj_target` / `v_proj_target`
>    （官方命名；非 `kt_proj/vt_proj`）。bias 跟随 `attention_bias`。
> 2. **融合参数**：模型顶层单个 Parameter `layer_fusion_weights ∈ R^{D×T}`
>    （非子模块 `layer_fusion.fuse_weights`）。前向：
>    `fusion_probs = softmax(layer_fusion_weights, dim=1)`；
>    `fused = einsum("bsth,dt->bsdh", target_hidden.view(b,s,T,h), fusion_probs)`。
> 3. **融合 RMSNorm 是共享的**：对 `[b,s,D,h]` 的 fused 整体施加**一个** `hidden_norm`
>    （即复用 DFlash 已有的 `hidden_norm` 权重键，形状 [h]），**不是**每 draft 层独立 norm。
>    DCore 相对 DFlash 删除的只有 `fc`。
> 4. **融合初始化**：全 0 后，每个 draft 层 d 在目标层索引
>    `t = min(T-1, int(d/D*T))` 处置 2.0 —— softmax 后每个 draft 层偏向一个
>    逐层递进的目标层（非均匀初始化）。
> 5. **默认目标层**（Qwen3-8B, 36 层）：`[1, 5, 9, 13, 17, 21, 25, 29, 33]`，
>    即 9 个 id 在 `[1, n-3]` 均匀分布（起点 1，非 2）。
> 6. **block_size 默认 8**，对齐 Budget8 检查点（mask_token_id=151669，全词表 151936，
>    冻结目标 embed/LM head —— 与仓库全词表路径一致）；Budget16 复现传 `--block-size 16`。
> 7. 官方检查点保存的键集合 = `layers.*`（含 `k_proj_target/v_proj_target`）、
>    `norm.weight`、`hidden_norm.weight`、`layer_fusion_weights`；
>    embed/lm_head 运行时取自目标模型 —— 与本仓库 `_keys_to_ignore_on_save` 约定吻合。

### 4.1 M1 — `AdaptiveLayerFusion` 模块（新文件 `models/dcore/model_definitions.py`）

```python
class AdaptiveLayerFusion(nn.Module):
    """DFlare 式逐 draft 层目标隐状态融合（arXiv:2606.02091 式 3）。"""

    def __init__(self, num_draft_layers: int, num_target_layers: int,
                 hidden_size: int, rms_norm_eps: float):
        super().__init__()
        # W_fuse ∈ R^{D×T}；零初始化 → softmax 后为均匀权重
        self.fuse_weights = nn.Parameter(
            torch.zeros(num_draft_layers, num_target_layers))
        self.fusion_norms = nn.ModuleList(
            Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)
            for _ in range(num_draft_layers))

    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]:
        # hidden_states: [1, L, T*d]（沿用现有数据管线的拼接布局）
        b, length, td = hidden_states.shape
        t = self.fuse_weights.shape[1]
        h = hidden_states.view(b, length, t, td // t)        # [1, L, T, d]
        alpha = torch.softmax(self.fuse_weights, dim=-1)      # [D, T]
        fused = torch.einsum("dt,blth->dblh", alpha, h)       # [D, 1, L, d]
        return [norm(fused[i]) for i, norm in enumerate(self.fusion_norms)]
```

要点：
- 输入沿用现有 `[1, L, T·d]` 拼接张量，仅 view 重排，**数据管线零改动**（T 由 `len(aux_hidden_state_layer_ids)` 推出）。
- 推理导出时 α 可随 checkpoint 预计算（§4.6）。

### 4.2 M2 — `Qwen3DCoreAttention`：异构 KV 投影

继承/改写 `Qwen3DFlashAttention`，仅改目标上下文的投影来源：

```python
class Qwen3DCoreAttention(Qwen3DFlashAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        kv_dim = config.num_key_value_heads * self.head_dim
        # 目标上下文专属投影（DFlare 式 4 的 Wt_K / Wt_V）
        self.kt_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.vt_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)

    # forward 中仅两行差异：
    #   k_ctx = self.kt_proj(target_hidden)   # 原: self.k_proj(target_hidden)
    #   v_ctx = self.vt_proj(target_hidden)   # 原: self.v_proj(target_hidden)
    # 其余（q_norm/k_norm、RoPE、拼接、attn_fn 分派）完全复用父类逻辑。
```

实现建议：把父类 `forward` 中 `k_ctx/v_ctx` 的两行提取成 `self._project_context(target_hidden)` 小钩子，再让 DCore 覆写——避免复制 60 行 forward（同时惠及未来变体）。这是对 `models/dflash/model_definitions.py` 唯一的侵入式修改，且语义不变（dflash 的钩子实现 = 原两行）。

`Qwen3DCoreDecoderLayer`：复制 `Qwen3DFlashDecoderLayer`，`self_attn = Qwen3DCoreAttention(...)`，其余不变。

### 4.3 M3 — `DCoreDraftModel` 与 `DCoreSpeculatorConfig`

`config.py`：

```python
@SpeculatorModelConfig.register("dcore")
class DCoreSpeculatorConfig(DFlashSpeculatorConfig):
    speculators_model_type: Literal["dcore"] = "dcore"
    architectures: list[str] = Field(
        default_factory=lambda: ["DCoreDraftModel"])
    # 其余字段全部继承（block_size / aux_hidden_state_layer_ids /
    # mask_token_id / sample_from_anchor / draft_vocab_size ...）
```

`core.py`（关键改动点对照 `DFlashDraftModel`）：

```python
@SpeculatorModel.register("dcore")
class DCoreDraftModel(DFlashDraftModel):
    config_class = DCoreSpeculatorConfig
    _no_split_modules = ["Qwen3DCoreDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        del self.fc, self.hidden_norm          # DCore 无共享 FC 融合
        self.layer_fusion = AdaptiveLayerFusion(
            num_draft_layers=len(self.layers),
            num_target_layers=len(self.target_layer_ids),
            hidden_size=config.transformer_layer_config.hidden_size,
            rms_norm_eps=config.transformer_layer_config.rms_norm_eps)

    def _make_decoder_layer(self, config, layer_idx):
        return Qwen3DCoreDecoderLayer(config.transformer_layer_config, layer_idx)
```

**父类重构（最小侵入）**：`DFlashDraftModel._backbone_forward` 中两处需要钩子化：

1. 融合：`core.py:418-420` 的
   ```python
   fc_output = self.fc(hidden_states); fc_output = self.hidden_norm(fc_output)
   ```
   提取为 `self._fuse_target_hidden(hidden_states) -> Tensor | list[Tensor]`（dflash 返回单张量）。
2. 逐层调用：`core.py:455-466` 的 `target_hidden=fc_output` 改为
   `target_hidden=fused if isinstance(fused, torch.Tensor) else fused[layer_idx]`。

这两处重构不改变 dflash/dflash2/dspark/xpress 的任何行为（默认路径返回单张量），由现有单测 + `test_dflash_vllm_parity` 保障。

训练接口直接继承：`get_trainer_kwargs`（dpace/gamma/max_anchors 等）、`forward`、`compute_metrics`（D-PACE 加权在 `models/dflash/metrics.py` 中按 `per_position_loss_weight` 分派）全部复用，**零改动**。

`from_training_args`：复用 `_build_base_config_kwargs("dcore", ...)`，无 dcore 专属超参（融合层数由 target_layer_ids 决定）。

检查点键管理：新增权重 `layer_fusion.fuse_weights`、`layer_fusion.fusion_norms.{i}.weight`、`layers.{i}.self_attn.{kt,vt}_proj.weight` 均为可训练、须保存；沿用父类 `_keys_to_ignore_on_save/load_missing` 规则（embed/verifier head 仍由 verifier 重建）。在 `tests/unit/models/test_checkpoint_key_ownership.py` 中补 dcore 条目。

### 4.4 M4 — 训练 CLI / 配置接线

| 文件 | 改动 |
|---|---|
| `src/speculators/models/__init__.py` | 导出 `DCoreDraftModel, DCoreSpeculatorConfig` |
| `train/config/schema.py` | `speculator_type` 描述加 `dcore`；`_DEFAULT_LOSS_FN["dcore"]="ce"`；`_DEFAULT_BLOCK_SIZE["dcore"]=16`；`_resolve_derived_defaults` 中 `is_dflash_family` 集合加 `"dcore"`，并让 `per_position_loss_weight` 默认 `"dpace"` 对 dcore 同样生效（现逻辑仅 `is_dflash`，改为 `speculator_type in {"dflash","dcore"}`） |
| `train/config/resolution.py` | `_ALGO_GROUP_USERS["dflash"]` 集合加 `"dcore"`（复用 DFlashArgs 组：block_size/decay_gamma/per_position_loss_weight/dpace_alpha/max_anchors） |
| `train/cli.py` | 无需改（模型类经 `SpeculatorModel.registry` 按 speculators_model_type 解析，cli.py:683） |

### 4.5 M5 — 9 目标层配方与数据管线

- `resolve_target_layer_ids`（`models/utils.py:101`）增加可选参数化：当 `speculator_type=dcore` 且未显式传 `--target-layer-ids` 时，默认在 `[2, n−3]` 均匀取 9 层（Qwen3-8B，n=36 → `2 6 10 14 18 21 25 29 33` 一类的等距序列）。实现上建议加 `num_aux_layers` 入参，由 cli 按算法传默认值，避免魔改全局默认（EAGLE3/dflash 仍 3 层）。
- 在线训练：`scripts/launch_vllm.py --target-layer-ids` 原样传 9 层（vLLM 的 `eagle_aux_hidden_state_layer_ids` 接受任意列表）；**传输与显存开销为 3 层方案的 3 倍**（Qwen3-8B bf16 下每 token 9×4096×2B = 72KB），需在长序列(8K)×高并发下实测带宽是否成为瓶颈；必要时降并发或先用 5 层配方过渡。
- 离线生成：`data_generation/offline.py` 走同一 `target_layer_ids`，磁盘占用同比 ×3，文档中注明。
- 兼容性校验：模型加载时校验 `len(aux_hidden_state_layer_ids) == layer_fusion.fuse_weights.shape[1]`，否则报错提示重新生成隐状态。

### 4.6 M6 — 转换器与导出

- `convert/dflash/` 模式新增 `convert/dcore/converter.py`（或扩展现有 DFlashConverter 支持 `num_target_layers` + 融合权重键的映射），支持从 AngelSlim 发布的 DFlare 检查点转入（DFlare 与 DCore 架构同构，仅训练目标不同，可直接作为冷启动/对照）。
- 导出产物：speculators 格式 config 中记录 `speculators_model_type="dcore"`、`aux_hidden_state_layer_ids`（9 层）、`block_size`；可选在导出时附带预计算的 `alpha = softmax(fuse_weights)`（serving 端免 softmax）。

### 4.7 M7 — 推理侧（vLLM）支持【独立工作流，可并行】

训练仓库内只能验证 **训练期 EAL/逐位置 acc**；端到端加速需要 serving 引擎支持 DCore 前向。两条路线：

1. **vLLM PR**（正路）：在 vLLM 的 DFlash 推理实现上增加 (a) per-layer 标量融合（读取预计算 α，推理时为 T 个隐状态的加权和，逐层不同）；(b) 异构 KV 投影（kt/vt_proj 权重加载 + ctx K/V 用其投影）。改动面与本仓库 M1/M2 对称。本仓库先行定义 checkpoint/config 约定，vLLM PR 按约定实现；`tests/integration/models/test_dflash_vllm_parity.py` 模式下新增 dcore 掩码/数值对齐测试。
2. **过渡方案**：用 SGLang（DCore 论文所用引擎，基于 SpecForge 生态）或 AngelSlim 的 DFlare serving 路径做外部评测，先验证训练收益。

### 4.8 M8 — 测试计划

| 层级 | 测试 | 验证点 |
|---|---|---|
| 单测 | `tests/unit/models/test_dcore_fusion.py` | α=softmax 行归一；零初始化=均匀平均；`[1,L,T·d]`→D 个 `[1,L,d]` 形状；梯度可达 fuse_weights |
| 单测 | `tests/unit/models/test_dcore_attention.py` | kt/vt_proj 生效（ctx 投影 ≠ 共享投影）；当 kt/vt 初始化为 k/v 的拷贝时输出与 DFlash 注意力逐位相等（守恒测试） |
| 单测 | `test_dcore_model_definitions.py` / config 注册 / `test_checkpoint_key_ownership.py` 扩展 | registry、save/load 往返、键归属 |
| 单测（已有） | `dpace_loss_decay` 已被 dflash 测试覆盖 | 复用，无需新增 |
| 集成 | `tests/integration/models/test_model_forward.py` 增加 dcore 条目 | eager/sdpa/flex 三后端一致 |
| e2e smoke | `tests/e2e/smoke/test_offline_training.py` 增加 dcore 变体 | 全管线可跑通 |
| 回归 | `tests/e2e/regression/` 增加 `test_dcore_qwen3_8b_*`（仿 `test_dflash2_qwen3_8b_sharegpt`） | 训练期 EAL 不低于同数据 dflash 基线（论文预期：τ 提升 ~0.2–0.4） |

### 4.9 M9 — 示例脚本与文档

- `examples/train/dcore_qwen3_8b_ultrachat_online_5k.sh`：以 `dflash_qwen3_8b_ultrachat_online_5k_bestpractices.sh` 为模板，改 `SPECULATOR_TYPE=dcore`、`TARGET_LAYER_IDS` 为 9 层、其余保持（dpace+ce+block16+5 层 draft）。
- `docs/user_guide/algorithms/dcore.md`：方法说明 + 与 dflash/dflash2 的选型建议；更新 `hidden_state_layer_id_selection.md` 增加 9 层指引。

---

## 5. 实施计划（阶段与验收）

| 阶段 | 内容 | 产出/验收标准 | 预估工作量 |
|---|---|---|---|
| **Phase 1** 模型核心 | M1 AdaptiveLayerFusion、M2 异构 KV、父类两处钩子化重构、M3 DCore 模型/配置与注册 | 单测全绿；dflash 族既有单测/集成测试无回归；守恒测试（kt=k,vt=v,α=均匀+禁 RMSNorm 时近似还原 DFlash 行为）通过 | 2–3 天 |
| **Phase 2** 训练接线 | M4 schema/resolution 接线、M5 9 层默认与校验、示例脚本 | `speculators train --speculator-type dcore --dry-run` 产出可加载 checkpoint；smoke 训练 loss 下降、EAL 指标正常输出 | 1–2 天 |
| **Phase 3** 验证 | M8 测试补齐 + 小规模对照实验（5k UltraChat，dcore vs dflash 同数据同预算） | 训练期 EAL：dcore ≥ dflash；逐位置 acc 曲线合理（后位增益明显） | 2–3 天（含训练等待） |
| **Phase 4** 推理与导出 | M6 转换器；M7 vLLM 支持（独立 PR，可并行启动） | checkpoint 可被 vLLM/SGLang 加载并跑 acceptance 评测（`examples/evaluate/` 流程）；端到端加速对照 dflash | 3–5 天（vLLM 侧不可控部分另计） |
| **Phase 5** 文档 | M9 算法文档、层选择指引、README features | 文档构建通过 | 0.5–1 天 |

依赖关系：P1→P2→P3 串行；P4 的 vLLM PR 可在 P1 定稿 checkpoint 约定后并行；P5 随时。

---

## 6. 训练配方（对照论文，供复现）

| 超参 | DCore 论文值 | 本仓库对应 |
|---|---|---|
| draft 层数 | 5 | `--num-layers 5`（dflash 默认） |
| 目标层数 T | 9（[2, n−3] 均匀） | `--target-layer-ids <9 ids>`（M5 默认化） |
| block_size / 预算 | 16（每预算独立 ckpt） | `--block-size 16`（默认） |
| 损失 | D-PACE 加权 CE | `--loss-fn ce --per-position-loss-weight dpace`（默认） |
| D-PACE α | 0.5 | `--dpace-alpha 0.5`（默认） |
| anchor | context-only，512 anchors/序列 | `sample_from_anchor=False`（默认）、`--max-anchors`（注意现最佳实践脚本用 3072） |
| 优化 | AdamW(0.9,0.999)、lr 6e-4、cosine→0、warmup 4%、clip 1.0、bf16 FSDP、3 epochs | 对应 CLI 旗标均已存在（注意 repo 默认 optimizer=muon，复现论文需 `--optimizer adamw`） |
| 数据 | Open-PerfectBlend prompts 由目标模型贪心重生成（4096 cap, seed 42） | `scripts/response_regeneration/` + `prepare-data` 流程 |
| 词表 | 冻结目标 LM head（全词表） | `--draft-vocab-size` 设为目标词表大小（现脚本常用 32000 缩减词表，复现论文须全词表） |

---

## 7. 风险与开放问题

1. **vLLM 推理支持是关键路径外部依赖**：训练收益（EAL）本仓库即可验证，但论文的加速比结论依赖 serving 引擎实现质量（CUDA graph 覆盖、α 预计算）。建议 P1 结束即冻结 checkpoint 格式并启动 vLLM 侧工作。
2. **RMSNorm/k_norm 细节与 AngelSlim 对齐**：融合 RMSNorm 是否逐层独立、k_norm 是否作用于 ctx keys，论文未写死；实现前花半天读 AngelSlim DFlare 源码核对，避免转换外部 checkpoint 时数值不一致。
3. **9 层隐状态的传输/存储开销**：在线训练 vLLM→trainer 带宽、离线数据磁盘均 ×3；8K 序列下需实测吞吐。缓解：先跑 5 层消融（DFlare 表 4 显示 5→9 层单调增益，5 层已有大部分收益）。
4. **D-PACE 增益分布**：论文显示增益集中在低熵任务、greedy-draft 设置；若业务负载高熵/采样 draft（Td=1），DCore 相对 DSpark 类 semi-AR 方案的优势会收窄（DCore 论文图 5/6）——这是选型层面的预期管理，不影响实现。
5. **sliding-window 目标模型**：若未来目标模型含滑窗层（如 gpt-oss），`fused[layer_idx]` 与滑窗 mask 组合已被现框架覆盖，但需在集成测试中加一个滑窗配置。
6. **缩减词表与 D-PACE 的交互**：CE 在缩减词表上计算时 q_i 为缩减词表内概率，存活概率代理略有偏差；论文配方是全词表。建议 dcore 默认全词表，缩减词表作为显式 opt-in 并在文档标注。

---

## 8. 参考

- DCore（对照研究）: [arXiv:2610.24993](https://arxiv.org/abs/2610.24993)，代码 github.com/ernie-research/DCore
- DFlare: [arXiv:2606.02091](https://arxiv.org/abs/2606.02091)，代码 [github.com/Tencent/AngelSlim](https://github.com/Tencent/AngelSlim)
- D-PACE: [arXiv:2605.18810](https://arxiv.org/abs/2605.18810)
- DFlash: arXiv:2602.06036
- 本仓库 DFlash 最佳实践: github.com/vllm-project/speculators issue #979
