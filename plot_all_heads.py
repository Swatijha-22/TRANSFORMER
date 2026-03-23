# plot_all_heads.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from extract_attention import get_attention

def plot_all_heads(tokens, attentions, layer=5):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(
        f'All 12 attention heads — Layer {layer+1}',
        fontsize=14, y=1.01
    )

    for head in range(12):
        ax = axes[head // 4][head % 4]
        attn = attentions[layer][0, head].numpy()

        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='YlOrRd',
            vmin=0, vmax=attn.max(),
            ax=ax,
            square=True,
            cbar=False,
            linewidths=0.2,
            linecolor='#eeeeee'
        )
        ax.set_title(f'Head {head+1}', fontsize=10)
        ax.tick_params(labelsize=7)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(f'all_heads_layer{layer+1}.png', dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Saved → all_heads_layer{layer+1}.png")


if __name__ == '__main__':
    sentence = "The river bank was flooded after the storm."
    tokens, attentions = get_attention(sentence)
    plot_all_heads(tokens, attentions, layer=5)