import argparse
import os
import time
from datetime import date
import string

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl
import uuid
import numpy as np
import scipy.sparse
from nltk.corpus import stopwords

from conv_pmf.model import ConvPMF
from conv_pmf.dataset import Amazon
from conv_pmf.data_loader import collate_fn
from extract_words.model import ExtractWords
from extract_words.dataset import EWAmazon
from common.dictionary import GloveDict6B
from common.word_embeds import GloveEmbeds
from common.util import show_elapsed_time
from common.topic_util import NPMIUtil


class Trainer(object):
    def __init__(
        self,
        conv_pmf_model,
        epsilon,
        train_loader,
        num_epoch,
        optimizer,
        val_loader,
        ew_model,
        ew_loader,
        ew_args,
        log_dir,
    ):
        self.conv_pmf_model = conv_pmf_model
        self.epsilon = epsilon
        self.train_loader = train_loader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.ew_model = ew_model
        self.ew_loader = ew_loader
        self.ew_args = ew_args
        self.writer = SummaryWriter(os.path.join(log_dir, "run"))

    def train_and_val(self):
        # initialize checkpoint directory
        checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # save initialized parameters
        torch.save(
            self.conv_pmf_model.state_dict(),
            os.path.join(checkpoint_dir, "initialized_checkpoint.pt"),
        )
        # train and eval loop
        for epoch_idx in range(1, self.num_epoch + 1):
            # 1. train epoch
            train_epoch_start = time.time()
            self.train_epoch(epoch_idx)
            train_epoch_end = time.time()
            show_elapsed_time(
                train_epoch_start, train_epoch_end, "train epoch {}".format(epoch_idx)
            )
            # 2. eval epoch
            val_epoch_start = time.time()
            self.val_epoch(epoch_idx)
            val_epoch_end = time.time()
            show_elapsed_time(
                val_epoch_start, val_epoch_end, "val epoch {}".format(epoch_idx)
            )
            # 3. calculate topic quality after training each epoch
            if epoch_idx >= 31:
                metric_epoch_start = time.time()
                self.metric_epoch(epoch_idx)
                metric_epoch_end = time.time()
                show_elapsed_time(
                    metric_epoch_start,
                    metric_epoch_end,
                    "metric epoch {}".format(epoch_idx),
                )
        # save final checkpoint
        torch.save(
            self.conv_pmf_model.state_dict(),
            os.path.join(checkpoint_dir, "checkpoint_final.pt"),
        )

        self.writer.close()

    def train_epoch(self, epoch_idx):
        self.conv_pmf_model.train()
        self.conv_pmf_model.cuda()
        batch_losses = []
        for batch_idx, (user_indices, docs, gt_ratings) in enumerate(self.train_loader):
            global_step = batch_idx + len(self.train_loader) * (epoch_idx - 1)
            user_indices = user_indices.to(device="cuda")
            docs = [doc.to(device="cuda") for doc in docs]
            gt_ratings = gt_ratings.to(device="cuda", dtype=torch.float32)
            self.optimizer.zero_grad()
            # forward
            estimate_ratings, total_entropy, factor_entropy = self.conv_pmf_model(
                user_indices, docs, with_entropy=True
            )
            mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
            if self.epsilon > 0.0:
                loss = mse + self.epsilon * total_entropy
            elif self.epsilon == 0.0:
                loss = mse
            else:
                raise ValueError("epsilon must be greater than or equal to 0.0")
            batch_losses.append(mse)
            # backward
            loss.backward()
            # model update
            torch.nn.utils.clip_grad_norm_(self.conv_pmf_model.parameters(), 1.0)
            self.optimizer.step()
            # log total avg entropy w.r.t. all factors of each batch
            self.writer.add_scalar(
                "Entropy/total_entropy",
                total_entropy.detach().cpu().numpy(),
                global_step,
            )
            # log avg entropy w.r.t. each factor of each batch
            # factor_entropy_np = factor_entropy.detach().cpu().numpy()
            # for factor, entropy in enumerate(factor_entropy_np):
            #     self.writer.add_scalar(
            #         "Entropy/factor_{}_entropy".format(factor), entropy, global_step,
            #     )
        # log avg loss of each epoch
        self.writer.add_scalar(
            "Loss/train",
            float(sum(batch_losses).detach().cpu().numpy() / len(batch_losses)),
            epoch_idx,
        )
        self.writer.flush()

    def val_epoch(self, epoch_idx):
        with torch.no_grad():
            self.conv_pmf_model.eval()
            self.conv_pmf_model.cuda()
            batch_losses = []
            for user_indices, docs, gt_ratings in self.val_loader:
                user_indices = user_indices.to(device="cuda")
                docs = [doc.to(device="cuda") for doc in docs]
                gt_ratings = gt_ratings.to(device="cuda", dtype=torch.float32)
                estimate_ratings = self.conv_pmf_model(
                    user_indices, docs, with_entropy=False
                )
                mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
                batch_losses.append(mse)
        # log avg loss of each epoch
        self.writer.add_scalar(
            "Loss/eval",
            float(sum(batch_losses).detach().cpu().numpy() / len(batch_losses)),
            epoch_idx,
        )
        self.writer.flush()

    def metric_epoch(self, epoch_idx):
        # 1. initialize trained embeddings and ew_model weights
        trained_embeds = self.conv_pmf_model.state_dict()["embedding.weight"]
        conv_weight = self.conv_pmf_model.state_dict()["conv1d.weight"]
        self.ew_model.eval()
        self.ew_model.cuda()
        self.ew_model.load_embeds(trained_embeds)
        self.ew_model.load_weight(conv_weight)

        # 2. get activation statistics
        # factor -> token -> (act_sum, act_cnt)
        factor2token2act_stat = {}
        for text_reviews in self.ew_loader:
            text_reviews = text_reviews.to(device="cuda")
            # [n_factor, batch_size, n_words]
            activations = self.ew_model(text_reviews)
            # for each factor
            for factor in range(self.ew_args["n_factor"]):
                token2act_stat = {}
                # for each review
                for review_idx in range(self.ew_args["ew_batch_size"]):
                    # [n_words,]
                    review_acts = activations[factor][review_idx]
                    # calculate entropy
                    prob_dist = torch.nn.functional.softmax(review_acts, dim=0)
                    entropy = -torch.sum(prob_dist * torch.log2(prob_dist))
                    # only condiser text reviews with small entropy
                    if entropy <= self.ew_args["ew_entropy_threshold"]:
                        # [n_words,]
                        review_acts = review_acts.detach().cpu().numpy()
                        # [n_words,]
                        review_tokens = text_reviews[review_idx].detach().cpu().numpy()
                        assert review_acts.shape == review_tokens.shape
                        # note that act_idx == token_idx
                        for act_idx in range(review_tokens.shape[0]):
                            act_val = review_acts[act_idx]
                            act_tokens = []
                            act_tokens.append(review_tokens[act_idx])
                            for offset in range(
                                1, (self.ew_args["window_size"] - 1) // 2 + 1
                            ):
                                if act_idx - offset >= 0:
                                    act_tokens.append(review_tokens[act_idx - offset])
                                if act_idx + offset < review_tokens.shape[0]:
                                    act_tokens.append(review_tokens[act_idx + offset])
                            for token in act_tokens:
                                if token in token2act_stat:
                                    token2act_stat[token][0] += act_val
                                    token2act_stat[token][1] += 1
                                else:
                                    token2act_stat[token] = [act_val, 1]
                factor2token2act_stat[factor] = token2act_stat

        # 3. extract words ordered by average activation value
        # for each factor, we first extract top 50 words
        NUM_TOPIC = 50
        factor2sorted_tokens_50 = {}
        factor2sorted_words_50 = {}
        dictionary = self.ew_args["dictionary"]
        # for topic kl divergence
        act_dist = torch.zeros(
            (self.ew_args["n_factor"], dictionary.vocab_size()),
            dtype=torch.float32,
            device="cuda",
        )
        for factor, token2act_stat in factor2token2act_stat.items():
            tokens = []
            avg_act_values = []
            for token, (act_sum, act_cnt) in token2act_stat.items():
                if act_cnt < self.ew_args["ew_least_act_num"]:
                    continue
                if (
                    token == dictionary.padding_idx()
                    or token == dictionary.unknown_idx()
                ):
                    continue
                tokens.append(token)
                avg_act_values.append(float(float(act_sum) / act_cnt))
                # for topic kl divergence
                act_dist[factor][token] = float(float(act_sum) / act_cnt)
            tokens = torch.tensor(tokens, dtype=torch.int32).to(device="cuda")
            avg_act_values = torch.tensor(avg_act_values, dtype=torch.float32).to(
                device="cuda"
            )
            indices = (
                torch.topk(avg_act_values, NUM_TOPIC).indices
                if NUM_TOPIC <= avg_act_values.shape[0]
                else torch.argsort(avg_act_values)
            )
            sorted_tokens_50 = list(tokens[indices].detach().cpu().numpy())
            factor2sorted_tokens_50[factor] = sorted_tokens_50
            sorted_words_50 = [dictionary.idx2word(token) for token in sorted_tokens_50]
            factor2sorted_words_50[factor] = sorted_words_50
        # select top k words for future use
        factor2sorted_tokens = {}
        factor2sorted_words = {}
        for factor, sorted_tokens_50 in factor2sorted_tokens_50.items():
            sorted_tokens = sorted_tokens_50[: self.ew_args["ew_k"]]
            factor2sorted_tokens[factor] = sorted_tokens
        for factor, sorted_words_50 in factor2sorted_words_50.items():
            sorted_words = sorted_words_50[: self.ew_args["ew_k"]]
            factor2sorted_words[factor] = sorted_words
        # save extracted words to text file
        words_dir = os.path.join(self.log_dir, "extracted_words")
        if not os.path.exists(words_dir):
            os.makedirs(words_dir)
        with open(
            os.path.join(words_dir, "factor2sorted_words_{}.txt".format(epoch_idx)),
            "w",
        ) as f:
            for factor, sorted_words in factor2sorted_words.items():
                f.write("factor {}: {}\n".format(factor, sorted_words))

        # 4. topic kl divergence
        act_dist = torch.nn.functional.softmax(act_dist, dim=-1)
        topic_kl_div_sum = torch.zeros(
            1, dtype=torch.float32, device=torch.device("cuda")
        )
        for i in range(self.ew_args["n_factor"]):
            for j in range(i + 1, self.ew_args["n_factor"]):
                # kl divergence between factor i and factor j
                topic_kl_div_sum += torch.sum(
                    act_dist[i] * (torch.log(act_dist[i]) - torch.log(act_dist[j]))
                )
        topic_kl_div = topic_kl_div_sum / (
            self.ew_args["n_factor"] * (self.ew_args["n_factor"] - 1) / 2
        )
        self.writer.add_scalar(
            "KL_divergence/topic_kl_div", topic_kl_div.detach().cpu().numpy(), epoch_idx
        )

        # 5. NPMI (Normalized (Pointwise) Mutual Information)
        token_cnt_mat = scipy.sparse.load_npz(self.ew_args["ew_token_cnt_mat_path"])
        npmi_util = NPMIUtil(token_cnt_mat)
        factor2npmi = npmi_util.compute_npmi(factor2sorted_tokens)
        self.writer.add_scalar(
            "NPMI/npmi_avg", np.mean(list(factor2npmi.values())), epoch_idx
        )

        # 6. word2vec similarity (trained_embeds)
        trained_embeds_np = trained_embeds.detach().cpu().numpy()
        original_embeds_np = self.ew_args["word_embeds"].embed_matrix()
        cos_sims_trained = []
        cos_sims_original = []
        for factor, sorted_tokens in factor2sorted_tokens.items():
            k = len(sorted_tokens)
            for i in range(k):
                for j in range(i + 1, k):
                    x_trained = trained_embeds_np[sorted_tokens[i]]
                    y_trained = trained_embeds_np[sorted_tokens[j]]
                    x_original = original_embeds_np[sorted_tokens[i]]
                    y_original = original_embeds_np[sorted_tokens[j]]
                    cos_sims_trained.append(
                        np.dot(x_trained, y_trained)
                        / (np.linalg.norm(x_trained) * np.linalg.norm(y_trained))
                    )
                    cos_sims_original.append(
                        np.dot(x_original, y_original)
                        / (np.linalg.norm(x_original) * np.linalg.norm(y_original))
                    )
        self.writer.add_scalar(
            "word2vec_similarity/trained_embeds", np.mean(cos_sims_trained), epoch_idx,
        )
        self.writer.add_scalar(
            "word2vec_similarity/original_embeds",
            np.mean(cos_sims_original),
            epoch_idx,
        )

        self.writer.flush()


def main():
    parser = argparse.ArgumentParser()
    # train and eval args
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--word_embeds_path", default="", type=str)
    parser.add_argument("--global_user_id2global_user_idx", default="", type=str)
    parser.add_argument("--global_item_id2global_item_idx", default="", type=str)
    parser.add_argument("--shuffle", default=True, type=bool)
    parser.add_argument("--train_batch_size", default=128, type=int)
    parser.add_argument("--val_batch_size", default=128, type=int)
    parser.add_argument("--num_epoch", default=20, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--n_word", default=128, type=int)
    parser.add_argument("--n_factor", default=32, type=int)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    # extract words args
    parser.add_argument("--ew_batch_size", default=128, type=int)
    parser.add_argument("--ew_entropy_threshold", type=float, default=float("inf"))
    parser.add_argument("--ew_least_act_num", default=50, type=int)
    parser.add_argument("--ew_k", default=10, type=int)
    parser.add_argument("--ew_token_cnt_mat_path", default="", type=str)
    # log args
    parser.add_argument("--log_dir", default="", type=str)
    parser.add_argument("--log_dir_level_2", default="", type=str)
    args = parser.parse_args()

    # initialize log_dir
    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    time_str = time.strftime("%H-%M-%S", time.localtime())
    datetime_str = date_str + "-" + time_str
    log_dir = os.path.join(
        "log",
        args.log_dir,
        args.log_dir_level_2,
        datetime_str + "-" + str(uuid.uuid4()),
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save all command args
    with open(os.path.join(log_dir, "hyper_params.txt"), "w") as f:
        f.write("dataset_path: {}\n".format(args.dataset_path))
        f.write("word_embeds_path: {}\n".format(args.word_embeds_path))
        f.write(
            "global_user_id2global_user_idx: {}\n".format(
                args.global_user_id2global_user_idx
            )
        )
        f.write(
            "global_item_id2global_item_idx: {}\n".format(
                args.global_item_id2global_item_idx
            )
        )
        f.write("train_batch_size: {}\n".format(args.train_batch_size))
        f.write("val_batch_size: {}\n".format(args.val_batch_size))
        f.write("num_epoch: {}\n".format(args.num_epoch))
        f.write("window_size: {}\n".format(args.window_size))
        f.write("n_word: {}\n".format(args.n_word))
        f.write("n_factor: {}\n".format(args.n_factor))
        f.write("epsilon: {}\n".format(args.epsilon))
        f.write("optimizer: {}\n".format("SGD"))
        f.write("lr: {}\n".format(args.lr))
        f.write("momentum: {}\n".format(args.momentum))
        f.write("weight_decay: {}\n".format(args.weight_decay))
        f.write("ew_batch_size: {}\n".format(args.ew_batch_size))
        f.write("ew_entropy_threshold: {}\n".format(args.ew_entropy_threshold))
        f.write("ew_least_act_num: {}\n".format(args.ew_least_act_num))
        f.write("ew_k: {}\n".format(args.ew_k))
        f.write("ew_token_cnt_mat_path: {}\n".format(args.ew_token_cnt_mat_path))

    dictionary = GloveDict6B(args.word_embeds_path)
    word_embeds = GloveEmbeds(args.word_embeds_path)
    with open(args.global_user_id2global_user_idx, "rb") as f:
        global_user_id2global_user_idx = pkl.load(f)
        global_num_user = len(global_user_id2global_user_idx)
    with open(args.global_item_id2global_item_idx, "rb") as f:
        global_item_id2global_item_idx = pkl.load(f)

    train_set = Amazon(
        args.dataset_path,
        "train",
        dictionary,
        args.n_word,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    )
    val_set = Amazon(
        args.dataset_path,
        "val",
        dictionary,
        args.n_word,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.train_batch_size,
        shuffle=args.shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.val_batch_size,
        shuffle=args.shuffle,
        collate_fn=collate_fn,
        drop_last=True,
    )
    conv_pmf_model = ConvPMF(
        global_num_user,
        args.n_factor,
        word_embeds,
        args.window_size,
        train_set.rating_mean(),
        train_set.rating_std(),
    )
    optimizer = torch.optim.SGD(
        conv_pmf_model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    ew_model = ExtractWords(args.n_factor, args.window_size, word_embeds.embed_dim())
    ew_model.eval()
    ew_model.cuda()
    ew_dataset = EWAmazon(args.dataset_path, dictionary, args.n_word)
    ew_loader = torch.utils.data.DataLoader(
        dataset=ew_dataset,
        batch_size=args.ew_batch_size,
        shuffle=False,
        drop_last=True,
    )
    ew_args = {
        "dictionary": dictionary,
        "word_embeds": word_embeds,
        "n_factor": args.n_factor,
        "window_size": args.window_size,
        "ew_batch_size": args.ew_batch_size,
        "ew_entropy_threshold": args.ew_entropy_threshold,
        "ew_least_act_num": args.ew_least_act_num,
        "ew_k": args.ew_k,
        "ew_token_cnt_mat_path": args.ew_token_cnt_mat_path,
    }
    trainer = Trainer(
        conv_pmf_model,
        args.epsilon,
        train_loader,
        args.num_epoch,
        optimizer,
        val_loader,
        ew_model,
        ew_loader,
        ew_args,
        log_dir,
    )
    trainer.train_and_val()


if __name__ == "__main__":
    main()
