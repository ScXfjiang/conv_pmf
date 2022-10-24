import random
import os

import argparse
import numpy as np
import pandas as pd
import json
import pickle as pkl


class Preprocessor(object):
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst

    def split_amazon(self, ratios=[0.7, 0.1, 0.2]):
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

    def generate_global_maps(self):
        """
        Those maps will not change during one train/val/test process
        """
        with open(self.src, "rb") as f:
            data = pd.DataFrame(
                index=np.arange(0, len(f.readlines())),
                columns=["user_id", "item_id", "rating", "text_review"],
            )
            f.seek(0)
            for idx, line in enumerate(f):
                js = json.loads(line)
                data.loc[idx] = [
                    str(js["reviewerID"]),
                    str(js["asin"]),
                    float(js["overall"]),
                    str(js["reviewText"]),
                ]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="", type=str)
    parser.add_argument("--dst", default="", type=str)
    args = parser.parse_args()

    if os.path.exists(args.dst):
        assert os.path.isdir(args.dst)
    else:
        os.makedirs(args.dst)

    proprocessor = Preprocessor(args.src, args.dst)
    proprocessor.split_amazon(ratios=[0.8, 0.1, 0.1])
    proprocessor.generate_global_maps()


if __name__ == "__main__":
    main()
