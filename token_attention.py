import matplotlib.pyplot as plt
import torch
from extract_attention import get_attention

def plot_token_attention(tokens, attentions, query_token, layer=5):
    # Find the index of the query token
    if query_token not in tokens:
        print(f"Token '{query_token}' not found. Available: {tokens}")
        return
    query_idx = tokens.index(query_token)

    # Average attention across all 12 heads for this layer
    # attentions[layer] shape: (1, 12, seq, seq)
    avg_attn = attentions[layer][0].mean(dim=0)  # (seq, seq)
    token_attn = avg_attn[query_idx].numpy()     # (seq,)

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ['#E86020' if t == query_token else '#378ADD' for t in tokens]
    bars = ax.bar(tokens, token_attn, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_title(
        f'Attention from "{query_token}" to all tokens (layer {layer+1}, avg of 12 heads)',
        fontsize=12
    )
    ax.set_xlabel('Token')
    ax.set_ylabel('Attention weight')
    plt.xticks(rotation=30, ha='right')

    for bar, val in zip(bars, token_attn):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.002,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=9
        )
    plt.tight_layout()
    plt.savefig(f'token_attn_{query_token}.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    sentence = "The river bank was flooded after the storm."
    tokens, attentions = get_attention(sentence)
    plot_token_attention(tokens, attentions, query_token='bank', layer=5)

