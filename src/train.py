import argparse
import os
import time
from datetime import date

import torch
from torch.utils.tensorboard import SummaryWriter
import pickle as pkl
import uuid

from conv_pmf.model import ConvPMF
from conv_pmf.dataset import Amazon
from conv_pmf.data_loader import collate_fn
from common.dictionary import GloveDict6B
from common.word_embeds import GloveEmbeds
from common.util import show_elapsed_time


class Trainer(object):
    def __init__(
        self, model, epsilon, train_loader, num_epoch, optimizer, val_loader, log_dir
    ):
        self.model = model
        self.epsilon = epsilon
        self.train_loader = train_loader
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.log_dir = log_dir
        self.writer = SummaryWriter(os.path.join(log_dir, "run"))

    def train_and_val(self):
        # initialize checkpoint directory
        checkpoint_dir = os.path.join(self.log_dir, "checkpoint")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # save initialized parameters
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "initialized_checkpoint.pt"),
        )
        # train and eval loop
        for epoch_idx in range(1, self.num_epoch + 1):
            # train epoch
            train_epoch_start = time.time()
            self.train_epoch(epoch_idx)
            train_epoch_end = time.time()
            show_elapsed_time(
                train_epoch_start, train_epoch_end, "Train epoch {}".format(epoch_idx)
            )
            # eval epoch
            val_epoch_start = time.time()
            self.val_epoch(epoch_idx)
            val_epoch_end = time.time()
            show_elapsed_time(
                val_epoch_start, val_epoch_end, "val epoch {}".format(epoch_idx)
            )
            # save checkpoint periodically
            if epoch_idx % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(epoch_idx)),
                )
        # save final checkpoint
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "checkpoint_final.pt"),
        )
        self.writer.close()

    def train_epoch(self, epoch_idx):
        self.model.train()
        self.model.cuda()
        batch_losses = []
        for batch_idx, (user_indices, docs, gt_ratings) in enumerate(self.train_loader):
            global_step = batch_idx + len(self.train_loader) * epoch_idx
            user_indices = user_indices.to(device="cuda")
            docs = [doc.to(device="cuda") for doc in docs]
            gt_ratings = gt_ratings.to(device="cuda", dtype=torch.float32)
            self.optimizer.zero_grad()
            # forward
            if self.epsilon > 0.0:
                estimate_ratings, entropy = self.model(
                    user_indices, docs, with_entropy=True
                )
                mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
                loss = mse + self.epsilon * entropy
                self.writer.add_scalar(
                    "Entropy/train", entropy.detach().cpu().numpy(), global_step,
                )
            elif self.epsilon == 0.0:
                estimate_ratings = self.model(user_indices, docs, with_entropy=False)
                mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
                loss = mse
            else:
                raise ValueError("epsilon must be greater than or equal to 0.0")
            batch_losses.append(mse)
            # backward
            loss.backward()
            # model update
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        self.writer.add_scalar(
            "Loss/train",
            float(sum(batch_losses).detach().cpu().numpy() / len(batch_losses)),
            epoch_idx,
        )
        self.writer.flush()

    def val_epoch(self, epoch_idx):
        with torch.no_grad():
            self.model.eval()
            self.model.cuda()
            batch_losses = []
            for user_indices, docs, gt_ratings in self.val_loader:
                user_indices = user_indices.to(device="cuda")
                docs = [doc.to(device="cuda") for doc in docs]
                gt_ratings = gt_ratings.to(device="cuda", dtype=torch.float32)
                estimate_ratings = self.model(user_indices, docs, with_entropy=False)
                mse = torch.nn.functional.mse_loss(estimate_ratings, gt_ratings)
                batch_losses.append(mse)
        self.writer.add_scalar(
            "Loss/eval",
            float(sum(batch_losses).detach().cpu().numpy() / len(batch_losses)),
            epoch_idx,
        )
        self.writer.flush()


def main():
    parser = argparse.ArgumentParser()
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
        args.epsilon,
        train_loader,
        args.num_epoch,
        optimizer,
        val_loader,
        log_dir,
    )
    trainer.train_and_val()


if __name__ == "__main__":
    main()
