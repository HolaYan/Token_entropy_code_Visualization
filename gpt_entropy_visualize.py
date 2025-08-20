import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from collections import defaultdict
import re
import os
from matplotlib.colors import LinearSegmentedColormap

# Set cache directories
os.environ["TRANSFORMERS_CACHE"] = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"
os.environ["HF_HOME"] = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"

# 🔥 新增：简单的activation收集器
def setup_activation_hooks(model, pattern=r"model\.layers\.\d+\.mlp$"):
    """一行代码设置activation hooks"""
    activities = defaultdict(list)
    hooks = []
    
    def mlp_hook_fn(module, input, output):
        # import pdb; pdb.set_trace()
        mlp_activations, auxiliary_output = output
        last_token_act = mlp_activations[:, -1, :]  # shape: [1, hidden_dim]
        # last_token_act = output[:, -1, :]  # shape: [1, hidden_dim]
        activities[module._hook_name].append(last_token_act.detach().cpu())
    
    # Setup MLP hooks
    compiled_mlp_pattern = re.compile(pattern)
    for name, module in model.named_modules():
        print(name)
        if compiled_mlp_pattern.match(name):
            module._hook_name = name
            hooks.append(module.register_forward_hook(mlp_hook_fn))
    
    return activities, hooks

def clean_hooks(hooks):
    """清理hooks"""
    for h in hooks:
        h.remove()

def analyze_token_entropy(model, tokenizer, text, max_new_tokens=10, collect_activations=False):
    """分析生成token的熵值 + 可选的activation收集 + logits收集"""
    
    # 🔥 新增：可选的activation收集
    activities, hooks = None, []
    if collect_activations:
        activities, hooks = setup_activation_hooks(model)
        activities.clear()
    
    # 编码输入
    inputs = tokenizer(text=text, return_tensors="pt").to(model.device)
    
    # 正常输出模式
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(f"生成的响应: {responses[0]}")
    print(f"输入文本: {text}")

    # 生成并获取scores
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的tokens和scores
    prompt_len = inputs['input_ids'].shape[1]
    gen_tokens = outputs.sequences[0][prompt_len:]
    scores = outputs.scores
    
    # 确保长度一致
    min_len = min(len(scores), len(gen_tokens))
    
    # 计算熵值和token字符串
    tokens = []
    entropies = []
    # 🔥 新增：保存logits用于分布可视化
    all_logits = []
    
    for i in range(min_len):
        # 获取当前步骤的logits
        logits = scores[i][0].cpu()  # [vocab_size]
        all_logits.append(logits)
        
        # 计算熵
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()
        entropies.append(entropy)
        
        # 获取token字符串
        token = tokenizer.decode([gen_tokens[i]], skip_special_tokens=False)
        tokens.append(token)
    
    # 🔥 新增：处理activations
    activations = None
    if collect_activations and activities:
        activations = {}
        for layer, acts in activities.items():
            # 处理MLP activations
            layer_acts = torch.cat(acts, dim=0)  # shape: [num_generated_tokens, hidden_dim]
            activations[layer] = layer_acts.cpu().float()
            
        clean_hooks(hooks)  # 清理
    
    # 🔥 处理attention数据
    if hasattr(outputs, 'attentions') and outputs.attentions:
        num_layers = len(outputs.attentions[0])  # 层数
        num_heads = outputs.attentions[0][0].shape[1]  # head数
        all_attn_matrices = {}

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                # layer_idx = 63   # 哪一层
                # head_idx = 7    # 哪个head

                attn_vectors = []
                lengths = []

                # 遍历每个生成步
                for i, step_attn in enumerate(outputs.attentions):
                    # step_attn: tuple of all layers
                    attn_tensor = step_attn[layer_idx]  # shape: [1, num_heads, 1, key_len]
                    # 取batch=0, head=head_idx, query=0
                    vec = attn_tensor[0, head_idx, 0, :]  # shape: [key_len]
                    attn_vectors.append(vec)
                    lengths.append(vec.shape[0])

                max_len = max(lengths)
                # pad每个向量到 max_len
                padded_vectors = []
                for vec in attn_vectors:
                    pad_size = max_len - vec.shape[0]
                    padded = torch.nn.functional.pad(vec, (0, pad_size), value=0)
                    padded_vectors.append(padded)
                # 堆叠成矩阵
                attn_matrix = torch.stack(padded_vectors, dim=0)  
                all_attn_matrices[f'layer_{layer_idx}_head_{head_idx}'] = attn_matrix
        # import pdb;pdb.set_trace()
        print("Padded attention matrix shape:", attn_matrix.shape)

    return tokens, entropies, activations, all_attn_matrices, all_logits

def compute_token_scores(tokens, entropies, activations, logits, attn_matrix, alpha=1.0, beta=1.0, delta=1.0):
    """
    计算每个token的四个指标和最终组合score
    """
    V = logits[0].size(0)
    max_token_entropy = np.log(V)

    # TokenEntropy
    token_entropy_norm = [e / max_token_entropy for e in entropies]

    # ActivationRise
    # 假设activations是 Dict[layer -> Tensor[num_tokens, hidden_dim]]
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
        uniformities.append(np.clip(uniformity,0,1))

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

def plot_token_scores(score_dict):
    """
    绘制每个token的指标和最终score
    """
    tokens = score_dict["tokens"]
    x = list(range(len(tokens)))
    fig, ax = plt.subplots(figsize=(14,5))

    ax.plot(x, score_dict["TokenEntropy"], label="TokenEntropy")
    ax.plot(x, score_dict["ActivationRise"], label="ActivationRise")
    ax.plot(x, score_dict["Uniformity"], label="Uniformity")
    ax.plot(x, score_dict["AttentionFocus"], label="AttentionFocus")
    ax.plot(x, score_dict["FinalScore"], label="FinalScore", linewidth=2, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_xlabel("Generated Tokens")
    ax.set_ylabel("Normalized Scores")
    ax.set_title("Per-token Indicators and Final Score")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_entropy_heatmap(tokens, entropies, max_cols=10, save_path=None):
    """绘制简洁的熵热力图（修复三角形问题）"""
    if not tokens:
        print("没有数据可绘制")
        return
    
    n_tokens = len(tokens)
    n_cols = min(max_cols, n_tokens)
    n_rows = (n_tokens + n_cols - 1) // n_cols
    
    fig, ax = plt.subplots(figsize=(n_cols * 1.2, n_rows * 0.8))
    
    # 归一化熵值用于颜色映射
    min_ent, max_ent = min(entropies), max(entropies)
    norm_entropies = [(e - min_ent) / (max_ent - min_ent) if max_ent > min_ent else 0.5 
                      for e in entropies]
    
    colormap = plt.cm.RdYlBu_r
    
    # 绘制每个token
    for i, (token, norm_ent, raw_ent) in enumerate(zip(tokens, norm_entropies, entropies)):
        row = i // n_cols
        col = i % n_cols
        x, y = col, n_rows - 1 - row
        
        # 绘制方块
        color = colormap(norm_ent)
        rect = Rectangle((x, y), 1, 1, facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        
        # 🔥 修复：移除三角形
        display_token = token.strip().replace('\n', '\\n')[:6]  # 去掉前后空格
        text_color = 'white' if norm_ent > 0.5 else 'black'
        
        ax.text(x + 0.5, y + 0.6, display_token, ha='center', va='center',
                fontsize=10, color=text_color, weight='bold')
        ax.text(x + 0.5, y + 0.3, f'{raw_ent:.2f}', ha='center', va='center',
                fontsize=8, color=text_color)
    
    # 设置图形
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=min_ent, vmax=max_ent))
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, shrink=0.8)
    cbar.set_label('Token Entropy', fontsize=12)
    
    plt.title(f'Token Entropy ({len(tokens)} tokens)', fontsize=14, pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"热力图保存到: {save_path}")
    
    plt.show()

def plot_per_token_layerwise_activations_with_entropy(
    activations, tokens, entropies=None, all_logits=None, tokenizer=None,
    save_dir="token_layer_activations"
):
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    layers = sorted(activations.keys())
    num_layers = len(layers)
    hidden_dim = activations[layers[0]].shape[1]
    num_tokens = activations[layers[0]].shape[0]

    for t in range(num_tokens):
        token_acts = []
        for layer in layers:
            act = activations[layer][t]
            token_acts.append(act.numpy())

        token_acts = np.stack(token_acts)  # [num_layers, hidden_dim]

        # 🔤 token 显示
        token_str = tokens[t].strip().replace('\n', '\\n')[:10]
        
        # 🔢 获取熵值
        ent = entropies[t] if entropies else None

        # 📊 获取top-3 logits预测（可选）
        top_logits_str = ""
        if all_logits and tokenizer:
            logits = all_logits[t]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_idxs = torch.topk(probs, 3)
            top_tokens = [tokenizer.decode([idx]) for idx in top_idxs]
            top_logits_str = ", ".join([f"{tok.strip()} ({p:.2f})" for tok, p in zip(top_tokens, top_probs)])

        # 绘图
        plt.figure(figsize=(10, 6))
        plt.imshow(token_acts, aspect='auto', cmap='seismic')
        plt.colorbar(label='Activation Value')
        plt.xlabel("Neurons")
        plt.ylabel("Layers")
        title = f"Token {t}: '{token_str}'"
        if ent:
            title += f" | Entropy: {ent:.2f}"
        if top_logits_str:
            title += f"\nTop logits: {top_logits_str}"
        plt.title(title, fontsize=12)
        plt.tight_layout()

        # 保存
        safe_token = re.sub(r'[^\w\-_.]', '_', token_str)
        fname = os.path.join(save_dir, f"token_{t}_{safe_token}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved: {fname}")

def plot_activations_overview_compact(activations, tokens, entropies=None, save_path=None):
    """
    更紧凑的activation概览版本，类似logits_overview的简洁风格
    """
    if not activations:
        print("没有activation数据")
        return
    
    layers = sorted(activations.keys())
    num_layers = len(layers)
    num_tokens = len(tokens)
    
    # 准备数据
    all_layer_acts = []
    for layer in layers:
        acts = activations[layer].numpy()  # [num_tokens, hidden_dim]
        all_layer_acts.append(acts)
    
    acts_matrix = np.stack(all_layer_acts)  # [num_layers, num_tokens, hidden_dim]
    
    # 创建左右两图布局（完全模仿logits_overview）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(4, num_tokens * 0.3)))
    
    # 左图：原始activation热力图
    # 取每层activation的L2 norm作为代表值
    acts_norm = np.linalg.norm(acts_matrix, axis=2)  # [num_layers, num_tokens]
    
    im1 = ax1.imshow(acts_norm.T, aspect='auto', cmap='viridis')  # 转置使token在y轴
    ax1.set_title(f'Activation Norms (L2)')
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Token Position')
    ax1.set_yticks(range(num_tokens))
    ax1.set_yticklabels([f"{i}: '{t.strip()[:6]}'" for i, t in enumerate(tokens)])
    ax1.set_xticks(range(num_layers))
    ax1.set_xticklabels([f"L{i}" for i in range(num_layers)])
    plt.colorbar(im1, ax=ax1)
    
    # 右图：activation变化率
    # 计算相邻层间的变化
    if num_layers > 1:
        acts_changes = np.diff(acts_norm, axis=0)  # [num_layers-1, num_tokens]
        im2 = ax2.imshow(acts_changes.T, aspect='auto', cmap='RdBu_r',
                        vmin=-np.abs(acts_changes).max(), vmax=np.abs(acts_changes).max())
        ax2.set_title(f'Layer-to-Layer Changes')
        ax2.set_xlabel('Layer Transition')
        ax2.set_ylabel('Token Position')
        ax2.set_yticks(range(num_tokens))
        ax2.set_yticklabels([f"{i}: '{t.strip()[:6]}'" for i, t in enumerate(tokens)])
        ax2.set_xticks(range(num_layers-1))
        ax2.set_xticklabels([f"L{i}→L{i+1}" for i in range(num_layers-1)], rotation=45)
        plt.colorbar(im2, ax=ax2)
    else:
        # 如果只有一层，显示activation分布
        acts_std = np.std(acts_matrix[0], axis=1)  # [num_tokens]
        ax2.bar(range(num_tokens), acts_std, color='skyblue', alpha=0.7)
        ax2.set_title('Activation Variability')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Std Dev')
        ax2.set_xticks(range(num_tokens))
        ax2.set_xticklabels([f"{i}: '{t.strip()[:4]}'" for i, t in enumerate(tokens)], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"紧凑activation概览图保存到: {save_path}")
    
    plt.show()

# 🔥 新增：简化版logits概览
def plot_logits_overview(tokens, all_logits, save_path=None):
    """简化的logits分布概览（热力图形式）"""
    if not all_logits:
        print("没有logits数据")
        return
    
    # 将所有logits堆叠成矩阵 [n_tokens, vocab_size]
    logits_matrix = torch.stack(all_logits)  # [n_tokens, vocab_size]
    
    # 只显示top-1000的词汇（避免图太大）
    top_k = min(1000, logits_matrix.shape[1])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, max(4, len(tokens) * 0.3)))
    
    # 左图：原始logits热力图
    top_logits = torch.topk(logits_matrix, top_k, dim=1)[0]  # [n_tokens, top_k]
    im1 = ax1.imshow(top_logits, aspect='auto', cmap='viridis')
    ax1.set_title(f'Raw Logits (Top {top_k})')
    ax1.set_xlabel('Vocab Rank')
    ax1.set_ylabel('Token Position')
    ax1.set_yticks(range(len(tokens)))
    ax1.set_yticklabels([f"{i}: '{t.strip()[:6]}'" for i, t in enumerate(tokens)])
    plt.colorbar(im1, ax=ax1)
    
    # 右图：概率分布热力图
    probs_matrix = torch.softmax(logits_matrix, dim=-1)
    top_probs = torch.topk(probs_matrix, top_k, dim=1)[0]
    im2 = ax2.imshow(torch.log(top_probs).numpy(), aspect='auto', cmap='plasma')
    ax2.set_title(f'Probabilities (Top {top_k})')
    ax2.set_xlabel('Vocab Rank')
    ax2.set_ylabel('Token Position')
    ax2.set_yticks(range(len(tokens)))
    ax2.set_yticklabels([f"{i}: '{t.strip()[:6]}'" for i, t in enumerate(tokens)])
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Logits概览图保存到: {save_path}")
    plt.show()


def analyze_high_logits(tokens, all_logits, tokenizer, logit_threshold=0.2, top_k=10):
    """
    找到logit较高的点，然后分析其前后5个token的logits选择
    
    Args:
        tokens: 生成的token列表
        all_logits: 对应的logits列表 
        tokenizer: tokenizer
        logit_threshold: logit阈值 (默认0.2)
        top_k: 打印top-k logits选择
    """
    import torch
    
    if len(tokens) == 0 or len(all_logits) == 0:
        print("Token序列或logits为空")
        return
    
    if len(tokens) != len(all_logits):
        print(f"警告: tokens长度({len(tokens)})与logits长度({len(all_logits)})不匹配")
        min_len = min(len(tokens), len(all_logits))
        tokens = tokens[:min_len]
        all_logits = all_logits[:min_len]
    
    # 找到高logit的位置
    high_logit_positions = []
    
    for i, logits in enumerate(all_logits):
        # 获取当前位置实际选择token的logit值
        if i < len(tokens):
            actual_token = tokens[i].strip()
            try:
                # 获取实际token的id
                token_ids = tokenizer.encode(actual_token, add_special_tokens=False)
                if token_ids:
                    actual_token_id = token_ids[0]
                    actual_logit = logits[actual_token_id].item()
                    
                    if actual_logit > logit_threshold:
                        high_logit_positions.append((i, actual_logit))
                        print(f"🎯 发现高logit: 位置{i}, token '{actual_token}', logit={actual_logit:.4f}")
            except Exception as e:
                print(f"处理位置{i}的token '{actual_token}'时出错: {e}")
                continue
    
    if not high_logit_positions:
        print(f"✅ 未发现logit > {logit_threshold} 的位置")
        return
    
    print(f"\n共发现 {len(high_logit_positions)} 个高logit位置")
    
    # 分析每个高logit位置前后5个token的logits
    for idx, (high_pos, high_logit_val) in enumerate(high_logit_positions):
        # 计算前后5个token的范围
        start_pos = max(0, high_pos - 5)
        end_pos = min(len(all_logits), high_pos + 6)  # +6是因为包含high_pos本身，前后各5个
        
        print(f"\n📊 高logit位置#{idx+1} - 分析位置{high_pos}前后5个token (位置 {start_pos} 到 {end_pos-1}) 的top-{top_k} logits选择:")
        print(f"    中心位置{high_pos}: '{tokens[high_pos]}' (logit={high_logit_val:.4f})")
        print("=" * 100)
        
        for pos in range(start_pos, end_pos):
            if pos >= len(all_logits) or pos >= len(tokens):
                continue
                
            logits = all_logits[pos]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, top_k)
            
            actual_token = tokens[pos].strip()
            
            # 标记这是否是中心的高logit位置
            position_marker = "🎯" if pos == high_pos else "  "
            relative_pos = pos - high_pos
            relative_marker = f"({relative_pos:+d})" if pos != high_pos else "(0)"
            
            print(f"\n{position_marker} 位置 {pos} {relative_marker}: 实际选择 '{actual_token}'")
            print("-" * 50)
            
            for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                try:
                    candidate_token = tokenizer.decode([idx]).strip()
                    # 检查是否是实际选择的token
                    actual_token_ids = tokenizer.encode(actual_token, add_special_tokens=False)
                    is_actual = "👉" if actual_token_ids and idx == actual_token_ids[0] else "  "
                    
                    # 获取对应的logit值
                    logit_val = logits[idx].item()
                    
                    print(f"    {rank+1:2d}. {is_actual} '{candidate_token:15s}' prob: {prob:.4f}, logit: {logit_val:.4f}")
                except Exception as e:
                    print(f"    {rank+1:2d}. [解码错误] prob: {prob:.4f}")
        
        print("=" * 100)

# 使用示例
if __name__ == "__main__":
    # 加载模型
    # model_path = "Qwen/Qwen2.5-32B-Instruct"
    # model_path = "Qwen/Qwen2.5-7B"
    model_path = "openai/gpt-oss-20b"
    dir_path = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        # attn_implementation="eager",
        device_map="auto",
        cache_dir=dir_path,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
    )

    # 分析文本 Find all c in Z_3 such that Z_3[x]\/(x^2 + c) is a field.\n\n A. 0\n B. 1\n C. 2\n D. 3
    text = "Please answer following question shortly.\n\nFind all c in Z_3 such that Z_3[x]\/(x^2 + c) is a field.\n\n A. 0\n B. 1\n C. 2\n D. 3"
    
    
    # 完整分析（熵 + activation + logits）
    tokens, entropies, all_acts, all_attn, logits = analyze_token_entropy(
        model, tokenizer, text, max_new_tokens=1024, 
        collect_activations=True
    )
    
    # 打印结果
    print("生成的tokens和熵值:")
    for token, entropy in zip(tokens, entropies):
        print(f"'{token.strip()}' -> {entropy:.4f}")
    
    print(f"\n平均熵: {np.mean(entropies):.4f}")
    print(f"熵范围: {np.min(entropies):.4f} ~ {np.max(entropies):.4f}")
    
    # 绘制所有图
    plot_entropy_heatmap(tokens, entropies, save_path="entropy.png")
    
    print(f"Collected: {list(all_acts.keys())}")

    # 分析的不是很好（重复位置找的不够好）
    # analyze_repetition_logits(tokens, logits, tokenizer, n_gram=5, top_k=10)

    # analyze_high_logits(tokens, logits, tokenizer, logit_threshold=0.2, top_k=10)
    
    # ✅ 7) 选择 base 层（第一层）
    base_layer = sorted(all_acts.keys())[0]
    base = all_acts[base_layer]

    # ✅ 8) 与 base 做差 & 可视化
    num_layers = len(all_acts)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 2 * num_layers))

    if num_layers == 1:
        axes = [axes]

    for ax, (layer, acts) in zip(axes, sorted(all_acts.items())):
        diff = acts - base  # 逐层 - base

        max_abs = max(abs(diff.min().item()), abs(diff.max().item()))
        im = ax.imshow(
            diff.numpy(),
            aspect='auto',
            cmap='seismic',
            vmin=-max_abs,
            vmax=max_abs,
            interpolation='nearest'
        )
        ax.set_title(f"{layer} vs {base_layer}")
        ax.set_xlabel("Neurons")
        ax.set_ylabel("Time steps")
        fig.colorbar(im, ax=ax)

    fig.suptitle("Layer-wise difference relative to first layer", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

    fig.savefig("mlp_layers_diff_vs_first.png", dpi=300)
    print("Saved: mlp_layers_diff_vs_first.png")
    

    # 看每一个token的activation分布
    # plot_per_token_layerwise_activations_with_entropy(
    #     all_acts, tokens, entropies, logits, tokenizer,
    #     save_dir="token_layer_activations"
    # )

    # 🔥 新增：紧凑的activation概览
    plot_activations_overview_compact(
        all_acts, tokens, entropies, save_path="activation_overview.png"
    )
    # 🔥 新增：绘制logits分布
    print(f"收集了{len(logits)}个token的logits分布")
    
    plot_logits_overview(tokens, logits, save_path="logits_overview.png")  # 概览热力图

    for key in all_attn:
        attentions = all_attn[key]
        print("Visualize token attention patterns...")
        heatmap = attentions.cpu().float().numpy()
        cmap=LinearSegmentedColormap.from_list("white_red", ["white", "red"])

        plt.figure(figsize=(12, 6))
        plt.imshow(heatmap, aspect="auto", cmap=cmap)
        plt.colorbar(label="Attention weight")

        # # x轴 index
        # plt.xticks(
        #     ticks=np.arange(heatmap.shape[1]),
        #     labels=np.arange(heatmap.shape[1]),
        #     rotation=90,
        #     fontsize=6
        # )

        # # y轴 index
        # plt.yticks(
        #     ticks=np.arange(heatmap.shape[0]),
        #     labels=np.arange(heatmap.shape[0])
        # )

        plt.xlabel("Key Token Index")
        plt.ylabel("Generated Token Index")
        plt.title("Attention Heatmap (Index Only)")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"{key}_attention_heatmap.png", dpi=300)
        plt.close()