import torch

def rowwise_recall(pred, target):
    assert pred.ndim == target.ndim
    assert pred.size()[:-1] == target.size()[:-1]

    # matches: boolean tensor of shape (..., K, L) where matches[..., k, l] is True if preds[..., k] == targets[..., l]
    matches = (pred.unsqueeze(-1) == target.unsqueeze(-2))

    # rowwise_isin: boolean tensor of shape (N, K) where rowwise_isin[..., k] is torch.isin(tensor_1[..., k], target_tensor[...])
    rowwise_isin = torch.sum(matches, dim=-1, dtype=torch.bool)

    # rowwise_recall: tensor of shape (N) where rowwise_recall[n] is the recall of preds[...] w.r.t. target[...]
    rowwise_recall = torch.sum(rowwise_isin, dim=-1).float() / target.size(-1)

    return rowwise_recall


def all_equal(l: list):
    """
    All elements are equal
    But some elements are more equal than others
    """
    return all(x == l[0] for x in l)
