# plot_heatmap.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from extract_attention import get_attention

def plot_head(tokens, attentions, layer=5, head=0):
    # Extract the attention matrix for this layer + head
    # Shape: (seq_len, seq_len)
    attn = attentions[layer][0, head].numpy()

    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',       # low=yellow, high=red
        vmin=0, vmax=attn.max(),
        ax=ax,
        square=True,
        linewidths=0.3,
        linecolor='#e8e8e8'
    )

    ax.set_title(
        f'Attention heatmap — Layer {layer+1}, Head {head+1}',
        fontsize=13, pad=12
    )
    ax.set_xlabel('Attending TO (key)', fontsize=11)
    ax.set_ylabel('Attending FROM (query)', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=150)
    print("Saved → attention_heatmap.png")
    plt.close()


if __name__ == '__main__':
    sentence = "The river bank was flooded after the storm."
    tokens, attentions = get_attention(sentence)
    plot_head(tokens, attentions, layer=5, head=0)

#red means more attention weight