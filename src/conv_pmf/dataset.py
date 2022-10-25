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
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.dictionary = dictionary
        self.n_token = n_token
        self.user_id2user_idx = global_user_id2global_user_idx
        self.item_id2item_idx = global_item_id2global_item_idx
        self.train_df = self.get_dataframe(train_path)
        self.val_df = self.get_dataframe(val_path)
        self.test_df = self.get_dataframe(test_path)
        self.item_id2doc = {}
        for item_id in self.item_id2item_idx.keys():
            df = self.train_df[self.train_df["item_id"] == item_id]
            if df.shape[0] == 0:
                doc = np.empty((0, self.n_token), dtype=np.int64)
            else:
                doc = np.array(list(df["tokens"]))
            self.item_id2doc[item_id] = doc

    def __getitem__(self, idx):
        if self.mode == "train":
            user_id, item_id, rating, tokens = self.train_df.iloc[idx]
            full_doc = self.item_id2doc[item_id]
            for idx in range(full_doc.shape[0]):
                if (full_doc[idx] == tokens).all():
                    doc = np.delete(full_doc, idx, axis=0)
                    break
            assert doc.shape[0] == full_doc.shape[0] - 1
        elif self.mode == "val":
            user_id, item_id, rating, _ = self.val_df.iloc[idx]
            doc = self.item_id2doc[item_id]
        elif self.mode == "test":
            user_id, item_id, rating, _ = self.test_df.iloc[idx]
            doc = self.item_id2doc[item_id]
        else:
            raise NotImplementedError
        return self.user_id2user_idx[user_id], doc, rating

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
                columns=["user_id", "item_id", "rating", "tokens"],
            )
            f.seek(0)
            for idx, line in enumerate(f):
                js = json.loads(line)
                df.at[idx, "user_id"] = str(js["reviewerID"])
                df.at[idx, "item_id"] = str(js["asin"])
                df.at[idx, "rating"] = float(js["overall"])
                df.at[idx, "tokens"] = self.tokenize(str(js["reviewText"]))
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

    def rating_mean(self):
        if self.mode == "train":
            df = self.train_df
        elif self.mode == "val":
            df = self.val_df
        elif self.mode == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        return df["rating"].mean()

    def rating_std(self):
        if self.mode == "train":
            df = self.train_df
        elif self.mode == "val":
            df = self.val_df
        elif self.mode == "test":
            df = self.test_df
        else:
            raise NotImplementedError
        return df["rating"].std()


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
