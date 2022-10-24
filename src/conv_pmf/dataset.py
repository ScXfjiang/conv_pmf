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
        path,
        dictionary,
        n_token,
        global_user_id2global_user_idx=None,
        global_item_id2global_item_idx=None,
    ):
        super(DatasetIf, self).__init__()

    def __getitem__(self, idx):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def rating_mean(self):
        raise NotImplementedError

    def rating_std(self):
        raise NotImplementedError


class Amazon(DatasetIf):
    """
    http://jmcauley.ucsd.edu/data/amazon/links.html
    """

    def __init__(
        self,
        path,
        dictionary,
        n_token,
        global_user_id2global_user_idx=None,
        global_item_id2global_item_idx=None,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )
        self.dictionary = dictionary
        self.n_token = n_token
        with open(path, "rb") as f:
            self.data = pd.DataFrame(
                index=np.arange(0, len(f.readlines())),
                columns=["user_id", "item_id", "rating", "text_review"],
            )
            f.seek(0)
            for idx, line in enumerate(f):
                js = json.loads(line)
                self.data.loc[idx] = [
                    str(js["reviewerID"]),
                    str(js["asin"]),
                    float(js["overall"]),
                    str(js["reviewText"]),
                ]
        if global_user_id2global_user_idx == None:
            user_ids = set(self.data["user_id"])
            self.user_id2user_idx = {id: idx for idx, id in enumerate(user_ids)}
        else:
            self.user_id2user_idx = global_user_id2global_user_idx
        if global_item_id2global_item_idx == None:
            item_ids = set(self.data["item_id"])
            self.item_id2item_idx = {id: idx for idx, id in enumerate(item_ids)}
        else:
            self.item_id2item_idx = global_item_id2global_item_idx
        self.item_idx2doc = {}
        for item_id, group in self.data.groupby("item_id"):
            doc = np.array(
                [
                    self.tokenize(text_review)
                    for text_review in list(group["text_review"])
                ],
                dtype=np.int32,
            )
            self.item_idx2doc[self.item_id2item_idx[item_id]] = doc
        self.data = self.data.drop("text_review", axis=1)

    def __getitem__(self, idx):
        user_id, item_id, rating = self.data.iloc[idx]
        user_idx = self.user_id2user_idx[user_id]
        doc = self.item_idx2doc[self.item_id2item_idx[item_id]]
        return user_idx, doc, rating

    def __len__(self):
        return self.data.shape[0]

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

    def rating_mean(self):
        return self.data["rating"].mean()

    def rating_std(self):
        return self.data["rating"].std()


class AmazonElectronics(Amazon):
    def __init__(
        self,
        path,
        dictionary,
        n_token,
        global_user_id2global_user_idx=None,
        global_item_id2global_item_idx=None,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )


class AmazonVideoGames(Amazon):
    def __init__(
        self,
        path,
        dictionary,
        n_token,
        global_user_id2global_user_idx=None,
        global_item_id2global_item_idx=None,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )


class AmazonGroceryAndGourmetFoods(Amazon):
    def __init__(
        self,
        path,
        dictionary,
        n_token,
        global_user_id2global_user_idx=None,
        global_item_id2global_item_idx=None,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
            global_user_id2global_user_idx,
            global_item_id2global_item_idx,
        )
