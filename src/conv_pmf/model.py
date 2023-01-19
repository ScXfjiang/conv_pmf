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
        """
        Args:
            user_indices: [batch_size,]
            docs: list of [num_review, num_word]
            with_entropy (bool, optional): entropy regularization
        """
        if with_entropy:
            self.forward_with_entropy(user_indices, docs)
        else:
            self.forward_without_entropy(user_indices, docs)

    def forward_with_entropy(self, user_indices, docs):
        """
        Args:
            user_indices: [batch_size,]
            docs: list of [num_review, num_word]
        """
        user_embeds = torch.index_select(self.w_user, 0, user_indices)
        item_embeds = []
        entropy_sum = 0.0
        num_entropy = 0
        for doc in docs:
            if doc.shape[0] != 0:
                # [num_review, embed_len, num_word]
                review_embeds = torch.permute(self.embedding(doc), (0, 2, 1))
                # [num_review, n_factor, num_word]
                feature_map = self.softmax_last_dim(self.conv1d(review_embeds))
                # [1, n_factor]
                item_embed = torch.mean(
                    torch.max(feature_map, dim=-1, keepdim=False).values,
                    dim=0,
                    keepdim=True,
                )
                # [doc_total_num_review, num_word]
                prob_dist = torch.reshape(feature_map, (-1, feature_map.shape[-1]))
                entropy_sum += -torch.sum(prob_dist * torch.log(prob_dist))
                num_entropy += prob_dist.shape[0]
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
        entropy = entropy_sum / num_entropy

        return estimate_ratings, entropy

    def forward_without_entropy(self, user_indices, docs):
        """
        Args:
            user_indices: [batch_size,]
            docs: list of [num_review, num_word]
        """
        user_embeds = torch.index_select(self.w_user, 0, user_indices)
        item_embeds = []
        for doc in docs:
            if doc.shape[0] != 0:
                # [num_review, embed_len, num_word]
                review_embeds = torch.permute(self.embedding(doc), (0, 2, 1))
                # [num_review, n_factor, num_word]
                feature_map = self.softmax_last_dim(self.conv1d(review_embeds))
                # [1, n_factor]
                item_embed = torch.mean(
                    torch.max(feature_map, dim=-1, keepdim=False).values,
                    dim=0,
                    keepdim=True,
                )
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

        return estimate_ratings
