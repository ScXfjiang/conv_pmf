import argparse
from datetime import date
import time
import os

import torch
import numpy as np
import pickle as pkl
import uuid

from extract_words.model import ExtractWords
from extract_words.dataset import get_dataset_type
from common.dictionary import get_dictionary_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="", type=str)
    parser.add_argument("--train_dataset_path", default="", type=str)
    parser.add_argument("--word_embeds_type", default="", type=str)
    parser.add_argument("--word_embeds_path", default="", type=str)
    parser.add_argument("--checkpoint_path", default="", type=str)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--n_word", default=128, type=int)
    parser.add_argument("--n_factor", default=32, type=int)
    parser.add_argument("--with_entropy", default="", type=str)
    parser.add_argument("--entropy_threshold", type=float, default=0.5)
    parser.add_argument("--least_act_num", default=200, type=int)
    parser.add_argument("--k", default=30, type=int)
    parser.add_argument("--use_cuda", default="", type=str)
    args = parser.parse_args()
    with_entropy = True if args.with_entropy == "True" else False
    use_cuda = True if args.use_cuda == "True" else False

    date_str = date.today().strftime("%b-%d-%Y")
    time_str = time.strftime("%H-%M-%S", time.localtime())
    log_dir = date_str + "-" + time_str + "-" + str(uuid.uuid4())
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    with open(os.path.join(log_dir, "hyper_params.txt"), "w") as f:
        f.write("dataset_type: {}\n".format(args.dataset_type))
        f.write("train_dataset_path: {}\n".format(args.train_dataset_path))
        f.write("word_embeds_type: {}\n".format(args.word_embeds_type))
        f.write("word_embeds_path: {}\n".format(args.word_embeds_path))
        f.write("checkpoint_path: {}\n".format(args.checkpoint_path))
        f.write("batch_size: {}\n".format(args.batch_size))
        f.write("window_size: {}\n".format(args.window_size))
        f.write("n_word: {}\n".format(args.n_word))
        f.write("n_factor: {}\n".format(args.n_factor))
        f.write("with_entropy: {}\n".format(with_entropy))
        f.write("entropy_threshold: {}\n".format(args.entropy_threshold))
        f.write("least_act_num: {}\n".format(args.least_act_num))
        f.write("k: {}\n".format(args.k))

    DictionaryT = get_dictionary_type(args.word_embeds_type)
    dictionary = DictionaryT(args.word_embeds_path)
    DatasetT = get_dataset_type(args.dataset_type)
    train_set = DatasetT(args.train_dataset_path, dictionary, args.n_word,)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=False, drop_last=True,
    )
    stat_dict = torch.load(args.checkpoint_path)
    trained_word_embeds = stat_dict["embedding.weight"]
    conv_weight = stat_dict["conv1d.weight"]
    model = ExtractWords(
        args.n_factor, args.window_size, trained_word_embeds, conv_weight
    )

    # 1. get activation statistics
    # factor -> token -> (act_sum, act_cnt)
    factor2token2act_stat = {}
    for factor in range(args.n_factor):
        factor2token2act_stat[factor] = {}
    entropy_list = []
    for _, reviews in enumerate(train_loader):
        if torch.cuda.is_available() and use_cuda:
            reviews = reviews.to(device="cuda")
        # [n_factor, batch_size, n_words]
        activations = model(reviews)
        for factor in range(args.n_factor):
            token2act_stat = {}
            for review_idx in range(args.batch_size):
                # [n_words,]
                cur_activations = activations[factor][review_idx]
                if with_entropy:
                    prob_dist = torch.nn.functional.softmax(cur_activations, dim=0)
                    entropy = -torch.sum(prob_dist * torch.log(prob_dist))
                    entropy_list.append(float(entropy.detach().cpu().numpy()))
                    if entropy > args.entropy_threshold:
                        continue
                cur_activations = cur_activations.detach().cpu().numpy()
                # [n_words,]
                cur_tokens = reviews[review_idx].detach().cpu().numpy()
                assert cur_activations.shape == cur_tokens.shape
                for act_idx in range(cur_tokens.shape[0]):
                    act_val = cur_activations[act_idx]
                    act_tokens = []
                    act_tokens.append(cur_tokens[act_idx])
                    for offset in range(1, (args.window_size - 1) // 2 + 1):
                        if act_idx - offset >= 0:
                            act_tokens.append(cur_tokens[act_idx - offset])
                        if act_idx + offset < cur_tokens.shape[0]:
                            act_tokens.append(cur_tokens[act_idx + offset])
                    for token in act_tokens:
                        if token in token2act_stat:
                            token2act_stat[token][0] += act_val
                            token2act_stat[token][1] += 1
                        else:
                            token2act_stat[token] = [act_val, 1]
            factor2token2act_stat[factor] = token2act_stat

    # 2. extract words by average activation value
    factor2sorted_tokens = {}
    factor2sorted_words = {}
    for factor, word2act_stat in factor2token2act_stat.items():
        token_list = []
        avg_act_value_list = []
        for word, (act_sum, act_cnt) in word2act_stat.items():
            if act_cnt < args.least_act_num:
                continue
            token_list.append(word)
            avg_act_value_list.append(float(float(act_sum) / act_cnt))
        tokens = torch.as_tensor(np.array(token_list))
        if torch.cuda.is_available() and use_cuda:
            tokens = tokens.to(device="cuda")
        avg_act_values = torch.as_tensor(np.array(avg_act_value_list))
        if torch.cuda.is_available() and use_cuda:
            avg_act_values = avg_act_values.to(device="cuda")
        indices = (
            torch.topk(avg_act_values, k=args.k).indices
            if args.k <= avg_act_values.shape[0]
            else torch.argsort(avg_act_values)
        )
        sorted_tokens = tokens[indices]
        factor2sorted_tokens[factor] = sorted_tokens
        sorted_words = [
            dictionary.idx2word(token)
            for token in list(sorted_tokens.detach().cpu().numpy())
        ]
        factor2sorted_words[factor] = sorted_words

    with open(os.path.join(log_dir, "factor2token2act_stat.pkl"), "wb") as f:
        pkl.dump(factor2token2act_stat, f)
    with open(os.path.join(log_dir, "factor2sorted_words.pkl"), "wb") as f:
        pkl.dump(factor2sorted_words, f)
    with open(os.path.join(log_dir, "factor2sorted_tokens.pkl"), "wb") as f:
        pkl.dump(factor2sorted_tokens, f)
    with open(os.path.join(log_dir, "factor2sorted_words.txt"), "w") as f:
        for factor, sorted_words in factor2sorted_words.items():
            f.write("factor {}: {}\n".format(factor, sorted_words))
    with open(os.path.join(log_dir, "entropy_stat.txt"), "w") as f:
        f.write("entropy mean: {}\n".format(np.mean(entropy_list)))
        f.write("entropy median: {}\n".format(np.median(entropy_list)))
        f.write("entropy min: {}\n".format(np.min(entropy_list)))
        f.write("entropy max: {}\n".format(np.max(entropy_list)))


if __name__ == "__main__":
    main()
