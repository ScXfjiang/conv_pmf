import json
import numpy as np
import pandas as pd
import torch
import torchtext


def get_dataset_type(type):
    if type == "amazon_electronics":
        DatasetT = AmazonElectronics
    elif type == "amazon_video_games":
        DatasetT = AmazonVideoGames
    elif type == "amazon_grocery_and_gourmet_foods":
        DatasetT = AmazonGroceryAndGourmetFoods
    else:
        raise NotImplementedError

    return DatasetT


class DatasetIf(torch.utils.data.Dataset):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        mode,
        dictionary,
        n_token,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    ):
        super(DatasetIf, self).__init__()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def train_set_rating_mean(self):
        raise NotImplementedError

    def train_set_rating_std(self):
        raise NotImplementedError


class Amazon(DatasetIf):
    """
    http://jmcauley.ucsd.edu/data/amazon/links.html
    """

    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        mode,
        dictionary,
        n_token,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    ):
        super().__init__(
            train_path,
            val_path,
            test_path,
            mode,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )
        self.train_df = self.get_dataframe(train_path)
        self.val_df = self.get_dataframe(val_path)
        self.test_df = self.get_dataframe(test_path)
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.dictionary = dictionary
        self.n_token = n_token
        self.user_id2user_idx = global_user_id2global_user_idx
        self.item_id2item_idx = global_item_id2global_item_idx

    def __getitem__(self, idx):
        if self.mode == "train":
            user_id, item_id, rating, _ = self.train_df.iloc[idx]
            user_idx = self.user_id2user_idx[user_id]
            # train set text reviews + train set ratings
            doc = np.array(
                [
                    self.tokenize(text_review)
                    for text_review in list(
                        self.train_df.drop(idx)
                        .groupby("item_id")
                        .get_group(item_id)["text_review"]
                    )
                ]
            )
            return user_idx, doc, rating
        elif self.mode == "val":
            user_id, item_id, rating, _ = self.val_df.iloc[idx]
            user_idx = self.user_id2user_idx[user_id]
            # train set text reviews + val set ratings
            doc = np.array(
                [
                    self.tokenize(text_review)
                    for text_review in list(
                        self.train_df.groupby("item_id").get_group(item_id)[
                            "text_review"
                        ]
                    )
                ]
            )
            return user_idx, doc, rating
        elif self.mode == "test":
            user_id, item_id, rating, _ = self.test_df.iloc[idx]
            user_idx = self.user_id2user_idx[user_id]
            # train set text reviews + test set ratings
            doc = np.array(
                [
                    self.tokenize(text_review)
                    for text_review in list(
                        self.train_df.groupby("item_id").get_group(item_id)[
                            "text_review"
                        ]
                    )
                ]
            )
            return user_idx, doc, rating
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode == "train":
            return self.train_df.shape[0]
        elif self.mode == "val":
            return self.val_df.shape[0]
        elif self.mode == "test":
            return self.test_df.shape[0]
        else:
            raise NotImplementedError

    def get_dataframe(self, path):
        with open(path, "rb") as f:
            df = pd.DataFrame(
                index=np.arange(0, len(f.readlines())),
                columns=["user_id", "item_id", "rating", "text_review"],
            )
            f.seek(0)
            for idx, line in enumerate(f):
                js = json.loads(line)
                df.loc[idx] = [
                    str(js["reviewerID"]),
                    str(js["asin"]),
                    float(js["overall"]),
                    str(js["reviewText"]),
                ]
        return df

    def tokenize(self, text_review):
        tokenizer = torchtext.data.get_tokenizer("basic_english")
        words = tokenizer(text_review)
        tokens = [self.dictionary.word2idx(word) for word in words]
        if self.n_token > len(tokens):
            tokens = tokens + [self.dictionary.padding_idx()] * (
                self.n_token - len(tokens)
            )
        else:
            tokens = tokens[: self.n_token]
        return tokens

    def train_set_rating_mean(self):
        return self.train_df["rating"].mean()

    def train_set_rating_std(self):
        return self.train_df["rating"].std()


class AmazonElectronics(Amazon):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        mode,
        dictionary,
        n_token,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    ):
        super().__init__(
            train_path,
            val_path,
            test_path,
            mode,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )


class AmazonVideoGames(Amazon):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        mode,
        dictionary,
        n_token,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    ):
        super().__init__(
            train_path,
            val_path,
            test_path,
            mode,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )


class AmazonGroceryAndGourmetFoods(Amazon):
    def __init__(
        self,
        train_path,
        val_path,
        test_path,
        mode,
        dictionary,
        n_token,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    ):
        super().__init__(
            train_path,
            val_path,
            test_path,
            mode,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )
