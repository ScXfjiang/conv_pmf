import argparse

import torch
import pickle as pkl

from conv_pmf.model import ConvPMF
from conv_pmf.dataset import get_dataset_type
from conv_pmf.data_loader import collate_fn
from common.dictionary import get_dictionary_type
from common.word_embeds import get_embeds_type


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="", type=str)
    parser.add_argument("--train_dataset_path", default="", type=str)
    parser.add_argument("--val_dataset_path", default="", type=str)
    parser.add_argument("--test_dataset_path", default="", type=str)
    parser.add_argument("--checkpoint_path", default="", type=str)
    parser.add_argument("--word_embeds_type", default="", type=str)
    parser.add_argument("--word_embeds_path", default="", type=str)
    parser.add_argument("--global_user_id2global_user_idx", default="", type=str)
    parser.add_argument("--global_item_id2global_item_idx", default="", type=str)
    parser.add_argument("--test_batch_size", default=128, type=int)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument("--n_word", default=128, type=int)
    parser.add_argument("--n_factor", default=32, type=int)
    args = parser.parse_args()

    DictionaryT = get_dictionary_type(args.word_embeds_type)
    dictionary = DictionaryT(args.word_embeds_path)
    EmbedT = get_embeds_type(args.word_embeds_type)
    word_embeds = EmbedT(args.word_embeds_path)
    with open(args.global_user_id2global_user_idx, "rb") as f:
        global_user_id2global_user_idx = pkl.load(f)
    with open(args.global_item_id2global_item_idx, "rb") as f:
        global_item_id2global_item_idx = pkl.load(f)
    DatasetT = get_dataset_type(args.dataset_type)
    test_set = DatasetT(
        args.train_dataset_path,
        args.val_dataset_path,
        args.test_dataset_path,
        "test",
        dictionary,
        args.n_word,
        global_user_id2global_user_idx,
        global_item_id2global_item_idx,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )
    with open(args.global_user_id2global_user_idx, "rb") as f:
        global_num_user = len(pkl.load(f))
    model = ConvPMF(
        global_num_user, args.n_factor, word_embeds, args.window_size, 0.0, 0.0,
    )
    model.load_state_dict(torch.load(args.checkpoint_path))
    with torch.no_grad():
        model.eval()
        model.cuda()
        losses = []
        for user_indices, docs, gt_ratings in test_loader:
            user_indices = user_indices.to(device="cuda")
            docs = [doc.to(device="cuda") for doc in docs]
            gt_ratings = gt_ratings.to(device="cuda", dtype=torch.float32)
            estimate_ratings = model(user_indices, docs, with_entropy=False)
            mse = torch.nn.functional.mse_loss(
                estimate_ratings, gt_ratings, reduction="sum"
            )
            losses.append(mse)
        final_mse = float(sum(losses) / len(test_set))
        print("Test MSE: {}".format(final_mse))


if __name__ == "__main__":
    main()
