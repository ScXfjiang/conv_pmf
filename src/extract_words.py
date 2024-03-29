import argparse
from datetime import date
import time
import os

import torch
import numpy as np
import scipy.sparse
import uuid

from extract_words.model import ExtractWords
from extract_words.dataset import EWAmazon
from common.dictionary import GloveDict6B
from common.word_embeds import GloveEmbeds
from common.topic_util import NPMIUtil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="", type=str)
    parser.add_argument("--word_embeds_path", default="", type=str)
    parser.add_argument("--checkpoint_path", default="", type=str)
    # conv_pmf args
    parser.add_argument("--n_factor", default=32, type=int)
    parser.add_argument("--n_word", default=128, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    # extract words args
    parser.add_argument("--strategy", default="", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--entropy_threshold", type=float, default=float("inf"))
    parser.add_argument("--least_act_num", default=50, type=int)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--ref_token_cnt_mat", default="", type=str)
    # log args
    parser.add_argument("--log_dir_level_1", default="", type=str)
    parser.add_argument("--log_dir_level_2", default="", type=str)
    args = parser.parse_args()
    assert args.strategy in ["all", "max"]

    # initialize log_dir
    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    time_str = time.strftime("%H-%M-%S", time.localtime())
    datetime_str = date_str + "-" + time_str
    log_dir = os.path.join(
        "log",
        args.log_dir_level_1,
        args.log_dir_level_2,
        datetime_str + "-" + str(uuid.uuid4()),
    )
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # save all command args
    with open(os.path.join(log_dir, "hyper_params.txt"), "w") as f:
        f.write("dataset_path: {}\n".format(args.dataset_path))
        f.write("word_embeds_path: {}\n".format(args.word_embeds_path))
        f.write("checkpoint_path: {}\n".format(args.checkpoint_path))
        f.write("n_factor: {}\n".format(args.n_factor))
        f.write("n_word: {}\n".format(args.n_word))
        f.write("window_size: {}\n".format(args.window_size))
        f.write("strategy: {}\n".format(args.strategy))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("entropy_threshold: {}\n".format(args.entropy_threshold))
        f.write("least_act_num: {}\n".format(args.least_act_num))
        f.write("ref_token_cnt_mat: {}\n".format(args.ref_token_cnt_mat))
        f.write("k: {}\n".format(args.k))

    # initialize dictionary
    dictionary = GloveDict6B(args.word_embeds_path)
    # original word embeds
    original_embeds = GloveEmbeds(args.word_embeds_path)
    # initialize extract word model
    ew_model = ExtractWords(
        args.n_factor, args.window_size, original_embeds.embed_dim()
    )
    ew_model.eval()
    ew_model.cuda()
    # load trained parameters
    stat_dict = torch.load(args.checkpoint_path)
    trained_embeds = stat_dict["embedding.weight"]
    conv_weight = stat_dict["conv1d.weight"]
    ew_model.load_embeds(trained_embeds)
    ew_model.load_weight(conv_weight)
    # initialize dataset and dataloader
    ew_dataset = EWAmazon(args.dataset_path, dictionary, args.n_word)
    ew_loader = torch.utils.data.DataLoader(
        dataset=ew_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )

    # 1. get activation statistics
    # factor -> token -> (act_sum, act_cnt)
    factor2token2act_stat = {}
    for text_reviews in ew_loader:
        text_reviews = text_reviews.to(device="cuda")
        # [n_factor, batch_size, n_words]
        activations = ew_model(text_reviews)
        # for each factor
        for factor in range(args.n_factor):
            token2act_stat = {}
            # for each review
            for review_idx in range(args.batch_size):
                # [n_words,]
                review_acts = activations[factor][review_idx].detach().cpu().numpy()
                # [n_words,]
                review_tokens = text_reviews[review_idx].detach().cpu().numpy()
                assert review_acts.shape == review_tokens.shape

                # note that act_idx == token_idx (padding="same" in conv op)
                if args.strategy == "all":
                    for act_idx in range(review_tokens.shape[0]):
                        act_val = review_acts[act_idx]
                        act_tokens = []
                        act_tokens.append(review_tokens[act_idx])
                        for offset in range(1, (args.window_size - 1) // 2 + 1):
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
                elif args.strategy == "max":
                    max_idx = np.argmax(review_acts)
                    act_val = review_acts[max_idx]
                    max_tokens = []
                    max_tokens.append(review_tokens[max_idx])
                    for offset in range(1, (args.window_size - 1) // 2 + 1):
                        if max_idx - offset >= 0:
                            max_tokens.append(review_tokens[max_idx - offset])
                        if max_idx + offset < review_tokens.shape[0]:
                            max_tokens.append(review_tokens[max_idx + offset])
                    for token in max_tokens:
                        if token in token2act_stat:
                            token2act_stat[token][0] += act_val
                            token2act_stat[token][1] += 1
                        else:
                            token2act_stat[token] = [act_val, 1]
                else:
                    raise NotImplementedError
            factor2token2act_stat[factor] = token2act_stat
    # 2. extract words ordered by average activation value
    # for each factor, we first extract top 50 words
    NUM_TOPIC = 50
    factor2sorted_tokens_50 = {}
    factor2sorted_words_50 = {}
    factor2sorted_act_values_50 = {}
    for factor, token2act_stat in factor2token2act_stat.items():
        tokens = []
        avg_act_values = []
        for token, (act_sum, act_cnt) in token2act_stat.items():
            if act_cnt < args.least_act_num:
                continue
            if token == dictionary.padding_idx() or token == dictionary.unknown_idx():
                continue
            tokens.append(token)
            avg_act_values.append(float(float(act_sum) / act_cnt))
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
        sorted_act_values_50 = list(avg_act_values[indices].detach().cpu().numpy())
        factor2sorted_act_values_50[factor] = sorted_act_values_50
    # select top k words for future use
    factor2sorted_tokens = {}
    factor2sorted_words = {}
    factor2sorted_act_values = {}
    for factor, sorted_tokens_50 in factor2sorted_tokens_50.items():
        sorted_tokens = sorted_tokens_50[: args.k]
        factor2sorted_tokens[factor] = sorted_tokens
    for factor, sorted_words_50 in factor2sorted_words_50.items():
        sorted_words = sorted_words_50[: args.k]
        factor2sorted_words[factor] = sorted_words
    for factor, sorted_act_values_50 in factor2sorted_act_values_50.items():
        sorted_act_values = sorted_act_values_50[: args.k]
        factor2sorted_act_values[factor] = sorted_act_values
    # save extracted words to text file
    words_dir = os.path.join(log_dir, "extracted_words")
    if not os.path.exists(words_dir):
        os.makedirs(words_dir)
    with open(os.path.join(words_dir, "factor2sorted_words.txt"), "w",) as f:
        for factor in factor2sorted_words.keys():
            f.write("factor {}: {}\n".format(factor, factor2sorted_words[factor]))
            f.write("factor {}: {}\n".format(factor, factor2sorted_act_values[factor]))

    # 3. word2vec similarity (original)
    original_embeds_np = original_embeds.embed_matrix()
    cos_sims_all = []
    for factor, sorted_tokens in factor2sorted_tokens.items():
        cos_sims_factor = []
        k = len(sorted_tokens)
        for i in range(k):
            for j in range(i + 1, k):
                x = original_embeds_np[sorted_tokens[i]]
                y = original_embeds_np[sorted_tokens[j]]
                w2v_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
                cos_sims_factor.append(w2v_sim)
                cos_sims_all.append(w2v_sim)
        with open(os.path.join(log_dir, "w2v_similarity.txt"), "a") as f:
            f.write("factor {}: {}\n".format(factor, np.mean(cos_sims_factor)))
    with open(os.path.join(log_dir, "w2v_similarity.txt"), "a") as f:
        f.write("overall w2v cosine similarity: {}\n".format(np.mean(cos_sims_all)))

    # 4. NPMI (Normalized (Pointwise) Mutual Information)
    ref_token_cnt_mat = scipy.sparse.load_npz(args.ref_token_cnt_mat)
    npmi_util = NPMIUtil(ref_token_cnt_mat)
    factor2npmi = npmi_util.compute_npmi(factor2sorted_tokens)
    for factor, npmi in factor2npmi.items():
        with open(os.path.join(log_dir, "npmi.txt"), "a") as f:
            f.write("factor {}: {}\n".format(factor, npmi))
    with open(os.path.join(log_dir, "npmi.txt"), "a") as f:
        f.write("overall NPMI: {}\n".format(np.mean(list(factor2npmi.values()))))


if __name__ == "__main__":
    main()
