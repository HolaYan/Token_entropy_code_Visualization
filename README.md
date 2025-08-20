token entropy以及attention以及activation等一系列提取+可视化等代码

有LLM也有多模态LLM的

log里面有打印出的qwen的模型框架

---

重复检测与可视化

为了检测模型生成中的重复现象，本仓库有以下可视化工具，并观察到在生成过程中存在重复的现象，表现为：

* **重复现象**：发生重复的地方，熵值相对提升（正常推理的起始位置也熵值较高），
* **激活值**：在重复片段中，激活值显著上升，并且集中在 **20层激活**。

### 1 可视化 token entropy 分布 

该工具用于展示每个 token 的 **entropy** 分布情况，帮助识别生成中熵值较低或异常的 token。通过可视化，我们发现重复位置的熵值显著增加。


### 2 可视化 activation 波动 

通过此工具，可以可视化每个 token 在生成过程中的 **activation** 波动。具体地：

* **左图**：每个 token 一行展示层级 **activation** 的 **L2 norm** 值，生成过程中的变化被以热图的形式展示。我们观察到，重复片段中的激活值显著增强，集中在 **20层激活**。
* **右图**：计算每个 token 在生成过程中的 **层间变化率**，显示重复片段的激活模式。


### 3 可视化 logits 分布

此工具通过热图的形式展示每个 token 的 **logits**，并打印出高熵处的 **logits** 计算与 **top-k** 候选 token。高熵部分往往与重复生成位置重合，有助于识别重复模式。


### 4 可视化注意力图 

该工具展示了挑选的某层的 **multi-head attention** 图，其中每个 attention head 的注意力分布可以清晰可见。通过这些图，我们发现重复的地方，注意力聚焦在少数几个 token 上，进一步证明了重复的现象。

---

### 计算 **token entropy**, **activation rise**, **uniformity**, **attention focus** 并生成 **score** 来监控坍缩

为了深入分析生成过程中可能发生的 **重复现象**，我们计算了每个 token 的四个关键指标：**token entropy**、**activation rise**、**uniformity** 和 **attention focus**。这些指标帮助我们定位生成过程中重复的部分，进而通过 **score** 来判断是否存在重复现象。以下是计算 **score** 的代码实现：

```python
import numpy as np
import torch

def compute_token_scores(tokens, entropies, activations, logits, attn_matrix, alpha=1.0, beta=1.0, delta=1.0):
    """
    计算每个token的四个指标和最终组合score
    """
    V = logits[0].size(0)
    max_token_entropy = np.log(V)

    # TokenEntropy
    token_entropy_norm = [e / max_token_entropy for e in entropies]

    # ActivationRise
    layer_name = list(activations.keys())[-1]
    acts = activations[layer_name]  # shape: [num_tokens, hidden_dim]
    norms = acts.norm(p=2, dim=-1)
    activation_rises = [0.0] + [(norms[i] - norms[i-1]).item() for i in range(1, len(norms))]
    min_act, max_act = min(activation_rises), max(activation_rises)
    activation_rise_norm = [(v - min_act) / (max_act - min_act + 1e-6) for v in activation_rises]

    # Uniformity
    uniformities = []
    log_uniform_prob = -np.log(V)
    for l in logits:
        probs = torch.softmax(l, dim=-1)
        log_probs = torch.log(probs + 1e-12)
        kl_div = (probs * (log_probs - log_uniform_prob)).sum().item()
        uniformity = 1 - kl_div
        uniformities.append(np.clip(uniformity, 0, 1))

    # Attention Focus
    max_attn_entropy = np.log(attn_matrix.size(1))
    attn_entropies = []
    for row in attn_matrix:
        norm_row = row / (row.sum() + 1e-12)
        entropy = -(norm_row * torch.log(norm_row + 1e-12)).sum().item()
        attn_entropies.append(entropy)
    attn_focus = [1 - (e / max_attn_entropy) for e in attn_entropies]

    # Final Score
    scores = []
    for i in range(len(tokens)):
        s = (
            token_entropy_norm[i]
            + alpha * activation_rise_norm[i]
            + beta * uniformities[i]
            + delta * attn_focus[i]
        )
        scores.append(s)

    # 打包返回
    result = {
        "tokens": tokens,
        "TokenEntropy": token_entropy_norm,
        "ActivationRise": activation_rise_norm,
        "Uniformity": uniformities,
        "AttentionFocus": attn_focus,
        "FinalScore": scores
    }
    return result
```

通过计算得出的 **final score**，我们可以有效地定位生成过程中可能出现的 **repetition** 部分。根据以下规则判断重复位置：

* **activation rise** 发生剧烈震荡，意味着模型在生成过程中遇到了 **认知冲突** 或 **推理问题**。
* **token entropy** 较低，表示模型对某些 token 的选择具有较强的偏好，这可能是 **重复生成** 或 **信息偏向** 的标志。

根据这些计算和可视化，我们可以清晰地识别出重复生成的区域，从而为 **动态温度调控** 提供有力支持，帮助减少重复并提高生成的多样性。
