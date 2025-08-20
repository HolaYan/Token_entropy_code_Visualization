import torch
import transformer_lens
from transformer_lens import HookedTransformer
# from transformer_lens.utilities import get_attention_patterns

import matplotlib.pyplot as plt

# Load GPT-2 small
model = HookedTransformer.from_pretrained("gpt2-small")

# Input prompt (repeated prefix)
prompt = "The quick brown fox jumps over the lazy dog. The quick"
tokens = model.to_tokens(prompt)
token_strs = model.to_str_tokens(tokens)

# Run the model and get cache
logits, cache = model.run_with_cache(tokens)



# For example, visualize Layer 5 Head 3
layer = 6
head = 7
# Get attention patterns for all layers
attn_patterns = attn_pattern = cache["attn", layer]
pattern = attn_patterns[:, head, :, :].detach().cpu().numpy()

# Plot attention heatmap
plt.figure(figsize=(8, 6))
plt.imshow(pattern[0], cmap="viridis")
plt.colorbar(label="Attention weight")
plt.xticks(range(len(token_strs)), token_strs, rotation=90)
plt.yticks(range(len(token_strs)), token_strs)
plt.title(f"Layer {layer} Head {head} Attention Pattern")
plt.tight_layout()
plt.show()
plt.savefig(f"layer_{layer}_head_{head}_attention_pattern.png")
