import torch
import torch.nn as nn
import numpy as np


class ExtractWords(nn.Module):
    def __init__(self, n_factor, window_size, trained_word_embeds, conv_weight):
        super(ExtractWords, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.as_tensor(trained_word_embeds), freeze=True,
        )
        self.conv1d = nn.Conv1d(
            in_channels=trained_word_embeds.shape[1],
            out_channels=n_factor,
            kernel_size=window_size,
            padding="same",
            bias=False,
        )
        self.softmax_last_dim = nn.Softmax(dim=-1)
        self.init_weight(conv_weight)
        self.n_factor = n_factor

    def init_weight(self, conv_weight):
        self.conv1d.weight = torch.nn.Parameter(conv_weight)

    def forward(self, reviews):
        # [batch_size, embedding_dim, n_words]
        review_embeds = torch.permute(self.embedding(reviews), (0, 2, 1))
        # [batch_size, n_factor, n_words]
        activations = self.softmax_last_dim(self.conv1d(review_embeds))
        # [n_factor, batch_size, n_words]
        activations = torch.permute(activations, (1, 0, 2))

        return activations
