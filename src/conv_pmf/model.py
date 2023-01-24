import torch
import torch.nn as nn
import math


class ConvPMF(nn.Module):
    def __init__(
        self, num_user, n_factor, word_embeds, window_size, rating_mean, rating_std,
    ):
        super(ConvPMF, self).__init__()
        # [num_user, n_factor]
        self.w_user = nn.parameter.Parameter(
            torch.empty((num_user, n_factor)), requires_grad=True,
        )
        # [voc_size, embed_length]
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.as_tensor(word_embeds.embed_matrix()), freeze=False,
        )
        self.conv1d = nn.Conv1d(
            in_channels=word_embeds.embed_dim(),
            out_channels=n_factor,
            kernel_size=window_size,
            padding="same",
            bias=False,  # only calculate activation, no need for bias
        )
        self.n_factor = n_factor
        self.tanh = nn.Tanh()
        self.softmax_last_dim = nn.Softmax(dim=-1)
        self.bias = nn.parameter.Parameter(torch.empty((1,)), requires_grad=True)
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
            return self.forward_with_entropy(user_indices, docs)
        else:
            return self.forward_without_entropy(user_indices, docs)
            # return self.forward_drop(user_indices, docs, quantile=0.5)
            # return self.forward_weighted_sum(user_indices, docs)

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
                feature_map = self.tanh(self.conv1d(review_embeds))
                # [1, n_factor]
                item_embed = torch.mean(
                    torch.max(feature_map, dim=-1, keepdim=False).values,
                    dim=0,
                    keepdim=True,
                )
                # [doc_total_num_review, num_word]
                prob_dist = self.softmax_last_dim(
                    torch.reshape(feature_map, (-1, feature_map.shape[-1]))
                )
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
                feature_map = self.tanh(self.conv1d(review_embeds))
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

    def forward_drop(self, user_indices, docs, quantile=0.5):
        """
        Args:
            user_indices: [batch_size,]
            docs: list of [num_review, num_word]
            quantile: keep quantile (e.g., 50%) reviews
        """
        user_embeds = torch.index_select(self.w_user, 0, user_indices)
        item_embeds = []
        for doc in docs:
            if doc.shape[0] != 0:
                # [num_review, embed_len, num_word]
                review_embeds = torch.permute(self.embedding(doc), (0, 2, 1))
                # [n_factor, num_review, num_word]
                feature_map = torch.permute(
                    self.tanh(self.conv1d(review_embeds)), (1, 0, 2)
                )

                # filter out reviews with high entropy
                n_factor, num_review, num_word = (
                    feature_map.shape[0],
                    feature_map.shape[1],
                    feature_map.shape[2],
                )
                # [n_factor, num_review, num_word]
                prob_dist = self.softmax_last_dim(feature_map)
                # [n_factor, num_review]
                entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=-1)
                # [n_factor, num_review * quantile]
                num_review = math.ceil(num_review * quantile)
                indices = torch.topk(
                    entropy, k=num_review, dim=-1, largest=False, sorted=False,
                ).indices
                # [n_factor, num_review * quantile, num_word]
                feature_map = torch.gather(
                    feature_map,
                    dim=1,
                    index=indices.unsqueeze(-1).expand(n_factor, num_review, num_word),
                )

                # [1, n_factor]
                item_embed = torch.mean(
                    torch.max(feature_map, dim=-1, keepdim=False).values,
                    dim=-1,
                    keepdim=False,
                ).unsqueeze(0)
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

    def forward_weighted_sum(self, user_indices, docs):
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
                # [n_factor, num_review, num_word]
                feature_map = torch.permute(
                    self.tanh(self.conv1d(review_embeds)), (1, 0, 2)
                )
                # [n_factor, num_review]
                max_values = torch.max(feature_map, dim=-1, keepdim=False).values
                # [n_factor, num_review, num_word]
                prob_dist = self.softmax_last_dim(feature_map)
                # [n_factor, num_review]
                entropy = -torch.sum(prob_dist * torch.log(prob_dist), dim=-1)
                z_score = (
                    entropy - torch.mean(entropy, dim=-1, keepdim=True)
                ) / torch.std(entropy, dim=-1, keepdim=True)
                # [n_factor, num_review]
                weights = self.softmax_last_dim(1 / self.tanh(z_score))
                # [1, n_factor]
                item_embed = torch.sum(max_values * weights, dim=-1).unsqueeze(0)
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
