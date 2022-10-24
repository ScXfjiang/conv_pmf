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
        path,
        dictionary,
        n_token,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
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
        self.corpus = [
            np.array(self.tokenize(text_review))
            for text_review in self.data["text_review"]
        ]

    def __getitem__(self, idx):
        return self.corpus[idx]

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


class AmazonElectronics(Amazon):
    def __init__(
        self,
        path,
        dictionary,
        n_token,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
        )


class AmazonVideoGames(Amazon):
    def __init__(
        self,
        path,
        dictionary,
        n_token,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
        )


class AmazonGroceryAndGourmetFoods(Amazon):
    def __init__(
        self,
        path,
        dictionary,
        n_token,
    ):
        super().__init__(
            path,
            dictionary,
            n_token,
        )
