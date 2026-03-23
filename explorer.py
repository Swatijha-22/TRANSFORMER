# explorer.py
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from extract_attention import get_attention

class AttentionExplorer:
    def __init__(self, tokens, attentions):
        self.tokens = tokens
        self.attentions = attentions
        self.layer = 0
        self.head = 0
        self.n_layers = len(attentions)
        self.n_heads = attentions[0].shape[1]

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.draw()
        # plt.show() commented out for headless environments
        # Uncomment on systems with display server

    def draw(self):
        self.ax.clear()
        attn = self.attentions[self.layer][0, self.head].numpy()

        sns.heatmap(
            attn,
            xticklabels=self.tokens,
            yticklabels=self.tokens,
            cmap='YlOrRd',
            vmin=0, vmax=attn.max(),
            ax=self.ax,
            square=True,
            linewidths=0.3,
            linecolor='#e8e8e8'
        )
        self.ax.set_title(
            f'Layer {self.layer+1} / Head {self.head+1}  '
            f'[← → change head | ↑ ↓ change layer]',
            fontsize=11
        )
        self.ax.set_xlabel('Attending TO (key)')
        self.ax.set_ylabel('Attending FROM (query)')
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(self.ax.get_yticklabels(), rotation=0)
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'right':
            self.head = (self.head + 1) % self.n_heads
        elif event.key == 'left':
            self.head = (self.head - 1) % self.n_heads
        elif event.key == 'up':
            self.layer = (self.layer + 1) % self.n_layers
        elif event.key == 'down':
            self.layer = (self.layer - 1) % self.n_layers
        self.draw()


if __name__ == '__main__':
    sentence = "The river bank was flooded after the storm."
    tokens, attentions = get_attention(sentence)
    AttentionExplorer(tokens, attentions)