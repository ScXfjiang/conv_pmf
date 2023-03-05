import torch
import torch.nn as nn
import numpy as np


class ExtractWords(nn.Module):
    def __init__(self, n_factor, window_size, embed_dim):
        super(ExtractWords, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim,
            out_channels=n_factor,
            kernel_size=window_size,
            padding="same",
            bias=False,
        )
        self.embedding = None
        self.tanh = nn.Tanh()
        self.n_factor = n_factor

    def load_embeds(self, embeds):
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.as_tensor(embeds), freeze=True,
        )

    def load_weight(self, conv_weight):
        self.conv1d.weight = torch.nn.Parameter(conv_weight)

    def forward(self, reviews):
        # [batch_size, embedding_dim, n_words]
        review_embeds = torch.permute(self.embedding(reviews), (0, 2, 1))
        # [batch_size, n_factor, n_words]
        activations = self.tanh(self.conv1d(review_embeds))
        # [n_factor, batch_size, n_words]
        activations = torch.permute(activations, (1, 0, 2))

        return activations
