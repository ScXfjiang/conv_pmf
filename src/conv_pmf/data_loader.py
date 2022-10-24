import torch

def collate_fn(batch):
    user_idx_list, doc_list, gt_rating_list = tuple(zip(*batch))
    user_indices = torch.as_tensor(user_idx_list, dtype=torch.int32)
    docs = []
    for doc in doc_list:
        doc = torch.as_tensor(doc, dtype=torch.int32)
        docs.append(doc)
    gt_ratings = torch.as_tensor(gt_rating_list, dtype=torch.float32)
    return user_indices, docs, gt_ratings