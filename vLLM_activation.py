import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info  # 🔥 注释掉这行，不使用process_vision_info
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from collections import defaultdict
import re
import os
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# Set cache directories
os.environ["TRANSFORMERS_CACHE"] = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"
os.environ["HF_HOME"] = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"

# 🔥 指定使用GPU 1（从0开始数的第1号GPU）
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 🔥 或者你也可以在运行脚本时指定：
# CUDA_VISIBLE_DEVICES=1 python your_script.py

# 🔥 激活收集器 - 专门收集最后一层MLP
def setup_last_layer_mlp_hook(model):
    """设置最后一层MLP的activation hook"""
    activities = []
    hooks = []
    
    def last_mlp_hook_fn(module, input, output):
        # 🔥 只保存最后一个token，立即转CPU以节省内存
        last_token_act = output[:, -1, :].detach().cpu()
        activities.append(last_token_act)
    
    # 找到最后一层的MLP - 根据模型结构应该是 model.layers.63.mlp
    last_layer_mlp = None
    for name, module in model.named_modules():
        print(name)
        if name == "model.layers.63.mlp":  # 最后一层（第63层）的MLP
            last_layer_mlp = module
            break
    # import pdb;pdb.set_trace()
    if last_layer_mlp is not None:
        hook = last_layer_mlp.register_forward_hook(last_mlp_hook_fn)
        hooks.append(hook)
        print(f"✅ 成功为最后一层MLP (model.layers.63.mlp) 设置hook")
    else:
        print("❌ 未找到最后一层MLP")
    
    return activities, hooks

def clean_hooks(hooks):
    """清理hooks"""
    for h in hooks:
        h.remove()

def analyze_vl_model_with_activations(model, processor, text, image_path=None, max_new_tokens=10, collect_activations=False):
    """分析视觉语言模型的生成 + activation收集"""
    
    # 🔥 设置activation收集
    activities, hooks = None, []
    if collect_activations:
        activities, hooks = setup_last_layer_mlp_hook(model)
        activities.clear()
    
    # 🔥 使用与工作代码相同的输入格式
    if image_path and os.path.exists(image_path):
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 使用工作代码的messages格式
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"file://{image_path}"}},
            ]},
        ]
    else:
        image = None
        # 纯文本输入
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": text}
            ]}
        ]
    
    # 🔥 使用与工作代码相同的处理方式
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    if image is not None:
        inputs = processor(text=text_input, images=image, return_tensors="pt")
    else:
        inputs = processor(text=text_input, return_tensors="pt")
    
    # 移动到设备 - 确保使用GPU 1
    device = torch.device("cuda:0")  # 这里是0因为CUDA_VISIBLE_DEVICES已经设置为1
    inputs = inputs.to(device)
    
    print(f"输入形状: {inputs['input_ids'].shape}")
    
    # 🔥 使用与工作代码相同的生成方式
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    # 🔥 使用与工作代码相同的输出处理方式
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    
    print(f"生成的响应: {response}")
    
    # 将token ID转换为字符串
    tokens = []
    for token_id in generated_ids:
        token_str = processor.tokenizer.decode(token_id, skip_special_tokens=True)
        tokens.append(token_str)
    
    # 🔥 处理activations
    activations = None
    if collect_activations and activities:
        print(f"收集到 {len(activities)} 个forward pass的activations")
        # 由于生成过程中每个token都会触发一次forward pass，
        # 我们需要提取对应生成token的activations
        if len(activities) > 0:
            # 取最后几个forward pass的activations（对应生成的tokens）
            num_gen_tokens = len(generated_ids)
            relevant_activations = activities[-num_gen_tokens:] if len(activities) >= num_gen_tokens else activities
            
            if relevant_activations:
                # 合并所有生成token的activations
                activations = torch.cat(relevant_activations, dim=0)  # shape: [num_generated_tokens, hidden_dim]
                print(f"最终activations形状: {activations.shape}")
    
    # 清理hooks
    if hooks:
        clean_hooks(hooks)
    
    return tokens, activations, response

def analyze_activations(activations):
    """分析activation数据"""
    if activations is None:
        print("没有收集到activation数据")
        return
    
    print(f"Activations形状: {activations.shape}")
    print(f"Mean activation: {activations.mean():.4f}")
    print(f"Std activation: {activations.std():.4f}")
    print(f"Min activation: {activations.min():.4f}")
    print(f"Max activation: {activations.max():.4f}")
    
    # 可选：分析每个token的activation统计
    print("\n每个生成token的activation统计:")
    for i in range(activations.shape[0]):
        token_act = activations[i]
        print(f"Token {i}: mean={token_act.mean():.4f}, std={token_act.std():.4f}")

def visualize_activations(activations, tokens, save_path="VL_activation_heatmap.png"):
    """可视化activation"""
    if activations is None:
        print("没有activation数据可视化")
        return
    
    plt.figure(figsize=(12, 8))
    
    # 创建热力图
    act_np = activations.float().numpy()
    
    # 如果维度太大，只显示前100个维度
    if act_np.shape[1] > 100:
        act_np = act_np[:, :100]
    
    plt.imshow(act_np.T, cmap='viridis', aspect='auto')
    plt.colorbar(label='Activation Value')
    plt.xlabel('Generated Token Position')
    plt.ylabel('Hidden Dimension')
    plt.title('Last Layer MLP Activations for Generated Tokens')
    
    # 添加token标签
    if len(tokens) <= 20:  # 只有token数量不太多时才显示
        plt.xticks(range(len(tokens)), [f"'{t}'" for t in tokens], rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"可视化已保存到: {save_path}")

# 使用示例
if __name__ == "__main__":
    # 🔥 检查GPU状态
    print("GPU状态检查:")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  内存使用: {memory_allocated:.2f}GB / {memory_total:.2f}GB")
        print(f"当前设备: {torch.cuda.current_device()}")
    else:
        print("CUDA不可用")
    
    # 加载模型
    model_path = "Qwen/Qwen2.5-VL-32B-Instruct"
    dir_path = "/gpfs/scratch/lh3862/project/VLLMreasoning/Model"
    
    print("🔥 正在加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": 0},  # 这里的0实际上是GPU 1（因为CUDA_VISIBLE_DEVICES="1"）
        cache_dir=dir_path,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        cache_dir=dir_path
    )
    
    print("✅ 模型加载完成")
    
    # 测试输入
    text = "Please reconstruct following images:"
    image_path = '/gpfs/scratch/lh3862/project/speedread_video/inference_codes/MMLU_5shot.png'
    
    print(f"🔥 开始分析模型...")
    print(f"文本输入: {text}")
    print(f"图像路径: {image_path}")
    
    print("\n=== 测试2: 收集activation ===")
    tokens2, activations2, response2 = analyze_vl_model_with_activations(
        model, processor, text, image_path=image_path, 
        max_new_tokens=1024, collect_activations=True
    )
    
    # 分析activation
    if activations2 is not None:
        analyze_activations(activations2)
        
        # 保存activations
        torch.save(activations2, 'last_layer_mlp_activations.pt')
        print("✅ Activations已保存到 last_layer_mlp_activations.pt")
        
        # 可视化activations
        visualize_activations(activations2, tokens2)
    else:
        print("❌ 未收集到activation数据")