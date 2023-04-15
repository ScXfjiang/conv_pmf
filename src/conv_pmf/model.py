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

    def forward(self, user_indices, docs, with_entropy=True):
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
        # entropy w.r.t. all factors
        total_entropy = torch.zeros(1, dtype=torch.float32, device=torch.device("cuda"))
        total_entropy_num = 0
        # entropy w.r.t. each factor
        factor_entropy = torch.zeros(
            self.n_factor, dtype=torch.float32, device=torch.device("cuda")
        )
        factor_entropy_num = 0
        # kl divergence
        batch_kl_div_sum = torch.zeros(
            1, dtype=torch.float32, device=torch.device("cuda")
        )
        batch_kl_div_num = 0
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
                # 1. calculate entropy
                # [n_factor, num_review, num_word]
                prob_dist = torch.permute(self.softmax_last_dim(feature_map), (1, 0, 2))
                # [n_factor, num_review]
                entropy = -torch.sum(
                    prob_dist * torch.log2(prob_dist), dim=-1, keepdim=False
                )
                # total entropy w.r.t. all factors
                total_entropy += torch.sum(entropy)
                total_entropy_num += entropy.shape[0] * entropy.shape[1]
                # entropy w.r.t. each factor
                factor_entropy += torch.sum(entropy, dim=-1, keepdim=False)
                factor_entropy_num += entropy.shape[1]
                # 2. calculate kl divergence
                # [n_factor, num_review * num_word]
                doc_kl_div_sum = torch.zeros(
                    1, dtype=torch.float32, device=torch.device("cuda")
                )
                for i in range(self.n_factor):
                    for j in range(i + 1, self.n_factor):
                        # doc_kl_div_sum += factor pairwise kl divergence w.r.t. a doc
                        doc_kl_div_sum += torch.mean(
                            torch.sum(
                                prob_dist[i]
                                * (torch.log(prob_dist[i]) - torch.log(prob_dist[j])),
                                axis=-1,
                            )
                        )
                doc_kl_div_avg = doc_kl_div_sum / (
                    self.n_factor * (self.n_factor - 1) / 2
                )
                batch_kl_div_sum += doc_kl_div_avg
                batch_kl_div_num += 1
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
        total_avg_entropy = total_entropy / total_entropy_num
        factor_avg_entropy = factor_entropy / factor_entropy_num
        batch_kl_div_avg = batch_kl_div_sum / batch_kl_div_num

        return estimate_ratings, total_avg_entropy, factor_avg_entropy, batch_kl_div_avg

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
                entropy = -torch.sum(prob_dist * torch.log2(prob_dist), dim=-1)
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
                entropy = -torch.sum(prob_dist * torch.log2(prob_dist), dim=-1)
                # min-max normalization
                # x_scaled = (x - x_min) / (x_max - x_min + epsilon) + offset
                # x_scaled is in [offset, 1 + offset]
                # 1/x_scaled is in [1/(1+offset), 1/offset]
                max = torch.max(entropy, dim=-1, keepdim=True).values
                min = torch.min(entropy, dim=-1, keepdim=True).values
                offset = 1e-1
                epsilon = 1e-5
                entropy_scaled = (entropy - min) / (max - min + epsilon) + offset
                # [n_factor, num_review]
                weights = self.softmax_last_dim(1 / entropy_scaled)
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
