import torch
import torch.nn as nn


class ConvPMF(nn.Module):
    def __init__(
        self, num_user, n_factor, word_embeds, window_size, rating_mean, rating_std,
    ):
        super(ConvPMF, self).__init__()
        self.w_user = nn.parameter.Parameter(
            torch.empty((num_user, n_factor)), requires_grad=True,
        )
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.as_tensor(word_embeds.embed_matrix()), freeze=False,
        )
        self.conv1d = nn.Conv1d(
            in_channels=word_embeds.embed_dim(),
            out_channels=n_factor,
            kernel_size=window_size,
            padding="same",
            bias=False,
        )
        self.n_factor = n_factor
        self.tanh = nn.Tanh()
        self.softmax_last_dim = nn.Softmax(dim=-1)
        self.bias = nn.parameter.Parameter(torch.empty((1,)), requires_grad=True,)
        self.rating_mean = rating_mean
        self.rating_std = rating_std
        self.init_weight()

    def init_weight(self):
        nn.init.uniform_(self.w_user, a=-self.rating_std, b=self.rating_std)
        nn.init.uniform_(self.conv1d.weight, a=-1.0, b=1.0)
        self.bias = torch.nn.Parameter(torch.tensor(self.rating_mean))

    def forward(self, user_indices, docs, with_entropy=False):
        user_embeds = torch.index_select(self.w_user, 0, user_indices)
        item_embeds = []
        if with_entropy:
            entropy_sum = 0.0
            num_entropy = 0
        for doc in docs:
            if doc.shape[0] != 0:
                review_embeds = torch.permute(self.embedding(doc), (0, 2, 1))
                feature_map = self.tanh(self.conv1d(review_embeds))
                item_embed = torch.mean(
                    torch.max(feature_map, dim=-1, keepdim=False).values,
                    dim=0,
                    keepdim=True,
                )
                if with_entropy:
                    prob_dist = self.softmax_last_dim(
                        torch.reshape(feature_map, (-1, feature_map.shape[-1]),)
                    )
                    doc_entropy = -torch.sum(prob_dist * torch.log(prob_dist))
                    entropy_sum += doc_entropy
                    doc_num_entropy = prob_dist.shape[0]
                    num_entropy += doc_num_entropy
            else:
                # deal with empty doc -> use self.bias as estimate rating
                item_embed = torch.zeros(
                    (1, self.n_factor),
                    dtype=torch.float32,
                    device=torch.device("cuda"),
                    requires_grad=False,
                )
            item_embeds.append(item_embed)
        item_embeds = torch.cat(item_embeds, dim=0)
        estimate_ratings = torch.sum(user_embeds * item_embeds, dim=-1) + self.bias
        if with_entropy:
            entropy = entropy_sum / num_entropy
            return estimate_ratings, entropy
        else:
            return estimate_ratings, None
