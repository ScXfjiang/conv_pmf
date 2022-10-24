import argparse
import os
import time
from datetime import date


import torch
import pickle as pkl
import matplotlib.pyplot as plt
import uuid

from conv_pmf.model import ConvPMF
from conv_pmf.dataset import get_dataset_type
from conv_pmf.data_loader import collate_fn
from common.dictionary import get_dictionary_type
from common.word_embeds import get_embeds_type


def show_elapsed_time(start, end, label=None):
    sec = end - start
    hour = int(sec // 3600)
    sec = sec - hour * 3600
    min = int(sec // 60)
    sec = sec - min * 60
    print(
        "{} elapsed time:\t {} hours {} mins {} seconds".format(label, hour, min, sec)
    )


class Trainer(object):
    def __init__(
        self,
        model,
        with_entropy,
        epsilon,
        train_loader,
        num_epoch,
        optimizer,
        val_loader,
        use_cuda,
        log_dir,
    ):
        self.model = model
        self.with_entropy = with_entropy
        self.epsilon = epsilon
        self.train_loader = train_loader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.use_cuda = use_cuda
        self.log_dir = log_dir
        self.train_loss_list = []
        self.val_loss_list = []

    def train_and_val(self):
        checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        for epoch_idx in range(1, self.num_epoch + 1):
            train_epoch_start = time.time()
            self.train_epoch()
            train_epoch_end = time.time()
            show_elapsed_time(
                train_epoch_start, train_epoch_end, "Train epoch {}".format(epoch_idx)
            )
            val_epoch_start = time.time()
            self.val_epoch()
            val_epoch_end = time.time()
            show_elapsed_time(
                val_epoch_start, val_epoch_end, "val epoch {}".format(epoch_idx)
            )
            if epoch_idx % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(epoch_idx)),
                )
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "checkpoint_final.pt"),
        )

        with open(os.path.join(self.log_dir, "train_loss"), "wb") as f:
            pkl.dump(self.train_loss_list, f)
        with open(os.path.join(self.log_dir, "val_loss"), "wb") as f:
            pkl.dump(self.val_loss_list, f)

        plt.plot(self.train_loss_list, label="train", linewidth=1)
        plt.plot(self.val_loss_list, label="val", linewidth=1)
        plt.legend(loc="upper right")
        plt.xlabel("num of epoch")
        plt.ylabel("MSE")
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, "mse_curve.pdf"))

    def train_epoch(self):
        self.model.train()
        if torch.cuda.is_available() and self.use_cuda:
            self.model.cuda()
        else:
            self.model.cpu()
        cur_losses = []
        for _, (user_indices, docs, gt_ratings) in enumerate(self.train_loader):
            if torch.cuda.is_available() and self.use_cuda:
                user_indices = user_indices.to(device="cuda")
                docs = [doc.to(device="cuda") for doc in docs]
                gt_ratings = gt_ratings.to(device="cuda")
            self.optimizer.zero_grad()
            estimate_ratings, entropy = self.model(
                user_indices, docs, with_entropy=self.with_entropy
            )
            gt_ratings = gt_ratings.to(torch.float32)
            mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
            if self.with_entropy:
                loss = mse + self.epsilon * entropy
            else:
                loss = mse
            cur_losses.append(mse)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.train_loss_list.append(float(sum(cur_losses) / len(cur_losses)))

    def val_epoch(self):
        with torch.no_grad():
            self.model.eval()
            if torch.cuda.is_available() and self.use_cuda:
                self.model.cuda()
            else:
                self.model.cpu()
            cur_losses = []
            for _, (user_indices, docs, gt_ratings) in enumerate(self.val_loader):
                if torch.cuda.is_available() and self.use_cuda:
                    user_indices = user_indices.to(device="cuda")
                    docs = [doc.to(device="cuda") for doc in docs]
                    gt_ratings = gt_ratings.to(device="cuda")
                estimate_ratings, _ = self.model(user_indices, docs, with_entropy=False)
                gt_ratings = gt_ratings.to(torch.float32)
                mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
                cur_losses.append(mse)
            self.val_loss_list.append(float(sum(cur_losses) / len(cur_losses)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="", type=str)
    parser.add_argument("--train_dataset_path", default="", type=str)
    parser.add_argument("--val_dataset_path", default="", type=str)
    parser.add_argument("--word_embeds_type", default="", type=str)
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
    parser.add_argument("--with_entropy", default="", type=str)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_cuda", default="", type=str)
    args = parser.parse_args()
    with_entropy = True if args.with_entropy == "True" else False
    use_cuda = True if args.use_cuda == "True" else False

    today = date.today()
    date_str = today.strftime("%b-%d-%Y")
    time_str = time.strftime("%H-%M-%S", time.localtime())
    datetime_str = date_str + "-" + time_str
    log_dir = os.path.join(args.dataset_type, datetime_str + "-" + str(uuid.uuid4()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "hyper_params.txt"), "w") as f:
        f.write("dataset_type: {}\n".format(args.dataset_type))
        f.write("train_dataset_path: {}\n".format(args.train_dataset_path))
        f.write("val_dataset_path: {}\n".format(args.val_dataset_path))
        f.write("word_embeds_type: {}\n".format(args.word_embeds_type))
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
        f.write("with_entropy: {}\n".format(with_entropy))
        f.write("epsilon: {}\n".format(args.epsilon))
        f.write("optimizer: {}\n".format("SGD"))
        f.write("lr: {}\n".format(args.lr))
        f.write("momentum: {}\n".format(args.momentum))
        f.write("weight_decay: {}\n".format(args.weight_decay))

    DictionaryT = get_dictionary_type(args.word_embeds_type)
    dictionary = DictionaryT(args.word_embeds_path)
    EmbedT = get_embeds_type(args.word_embeds_type)
    word_embeds = EmbedT(args.word_embeds_path)
    with open(args.global_user_id2global_user_idx, "rb") as f:
        global_user_id2global_user_idx = pkl.load(f)
        global_num_user = len(global_user_id2global_user_idx)
    with open(args.global_item_id2global_item_idx, "rb") as f:
        global_item_id2global_item_idx = pkl.load(f)

    DatasetT = get_dataset_type(args.dataset_type)
    train_set = DatasetT(
        args.train_dataset_path,
        dictionary,
        args.n_word,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    )
    val_set = DatasetT(
        args.val_dataset_path,
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
    model = ConvPMF(
        global_num_user,
        args.n_factor,
        word_embeds,
        args.window_size,
        train_set.rating_mean(),
        train_set.rating_std(),
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    trainer = Trainer(
        model,
        with_entropy,
        args.epsilon,
        train_loader,
        args.num_epoch,
        optimizer,
        val_loader,
        use_cuda,
        log_dir,
    )
    trainer.train_and_val()


if __name__ == "__main__":
    main()
