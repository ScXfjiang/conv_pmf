import random
import os

import argparse
import numpy as np
import scipy
import pandas as pd
import json
import pickle as pkl

from common.topic_util import gen_sparse_token_cnt_mat
from common.dictionary import GloveDict6B


class Preprocessor(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def text_preprocessing(self):
        """
        Text preprocessing.
        1. remove stopwords & punctuations
        2. downcasing
        """
        # TODO
        pass

    def split_amazon(self, ratios=[0.8, 0.1, 0.1]):
        """
        Split Amazon dataset into train/validation/test according to the given ratios.
        Amazon dataset link: http://jmcauley.ucsd.edu/data/amazon/links.html
        """
        assert len(ratios) == 3
        assert sum(ratios) == 1.0
        data = []
        with open(self.src, "rb") as f:
            for line in f:
                data.append(line)
        random.shuffle(data)

        train_size = int(len(data) * float(ratios[0]))
        val_size = int(len(data) * float(ratios[1]))
        with open(os.path.join(self.dst, "train.json"), "wb") as f:
            f.writelines(data[:train_size])
        with open(os.path.join(self.dst, "val.json"), "wb") as f:
            f.writelines(data[train_size : train_size + val_size])
        with open(os.path.join(self.dst, "test.json"), "wb") as f:
            f.writelines(data[train_size + val_size :])

    def gen_global_maps(self):
        """
        Generate global maps:
        1. global_user_id -> global_user_idx
        2. global_item_id -> global_item_idx
        Those maps will not change during one train/val/test process
        """
        with open(self.src, "rb") as f:
            data = pd.DataFrame(
                index=np.arange(0, len(f.readlines())), columns=["user_id", "item_id"],
            )
            f.seek(0)
            for idx, line in enumerate(f):
                js = json.loads(line)
                data.loc[idx] = [str(js["reviewerID"]), str(js["asin"])]
            global_user_ids = set(data["user_id"])
            global_item_ids = set(data["item_id"])
            global_user_id2global_user_idx = {
                id: idx for idx, id in enumerate(global_user_ids)
            }
            global_item_id2global_item_idx = {
                id: idx for idx, id in enumerate(global_item_ids)
            }
        with open(
            os.path.join(self.dst, "global_user_id2global_user_idx.pkl"), "wb"
        ) as f:
            pkl.dump(global_user_id2global_user_idx, f)
        with open(
            os.path.join(self.dst, "global_item_id2global_item_idx.pkl"), "wb"
        ) as f:
            pkl.dump(global_item_id2global_item_idx, f)

    def gen_token_cnt_mat(self, reference, word_embeds_path):
        """
        Generate reference to calculate NPMI.
        For now, we just use the whole Amazon dataset as the reference.
        This can be changed to a larger corpus, e.g., Wikipedia.
        Args:
            dataset_path: the whole Amazon dataset, train + val + test
            word_embeds_path: used for creating dictionary
        """
        token_cnt_mat = gen_sparse_token_cnt_mat(
            reference, GloveDict6B(word_embeds_path)
        )
        scipy.sparse.save_npz(
            os.path.join(self.dst, "token_cnt_mat.npz"), token_cnt_mat
        )


def main():
    parser = argparse.ArgumentParser()
    # the original Amazon dataset json file
    # e.g., reviews_Grocery_and_Gourmet_Food_5.json
    parser.add_argument("--src", default="", type=str)
    # the directory to store the processed data
    parser.add_argument("--dst", default="", type=str)
    # used to generate token_cnt_mat.npz for NPMI
    parser.add_argument("--reference", default="", type=str)
    # used to generate token_cnt_mat.npz for NPMI
    parser.add_argument("--word_embeds_path", default="", type=str)
    args = parser.parse_args()

    # create dst directory if not exists
    if os.path.exists(args.dst):
        assert os.path.isdir(args.dst)
    else:
        os.makedirs(args.dst)

    preprocessor = Preprocessor(args.src, args.dst)
    # 1. text preprocessing
    preprocessor.text_preprocessing()
    # 2. split the original Amazon dataset into train/val/test json files
    preprocessor.split_amazon(ratios=[0.8, 0.1, 0.1])
    # 3. generate global maps
    # global_user_id -> global_user_idx
    # global_item_id -> global_item_idx
    preprocessor.gen_global_maps()
    # 4. generate reference to calculate NPMI
    preprocessor.gen_token_cnt_mat(args.reference, args.word_embeds_path)


if __name__ == "__main__":
    main()
