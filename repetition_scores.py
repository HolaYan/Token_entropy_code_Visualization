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
        last_token_act = output[:, -1, :]  # shape: [1, hidden_dim]
        activities[module._hook_name].append(last_token_act.detach().cpu())
    
    # Setup MLP hooks
    compiled_mlp_pattern = re.compile(pattern)
    for name, module in model.named_modules():
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
    

# 使用示例
if __name__ == "__main__":
    # 加载模型
    # model_path = "Qwen/Qwen2.5-32B-Instruct"
    model_path = "Qwen/Qwen2.5-7B"
    dir_path = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        attn_implementation="eager",
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
    
    # 假设选一个attention矩阵
    attn_key = "layer_20_head_20"#list(all_attn.keys())[0]
    attn_matrix = all_attn[attn_key]

    # 计算分数
    scores = compute_token_scores(
        tokens,
        entropies,
        all_acts,
        logits,
        attn_matrix,
        alpha=1.0, beta=1.0, delta=1.0
    )

    # 可视化
    plot_token_scores(scores)