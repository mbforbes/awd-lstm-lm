# imports
# ---

# builtins
import code
from typing import Union

# 3rd party
import torch
from torch.autograd import Variable


# types
# ---

LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


# code
# ---

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz: int, args) -> LongTensor:
    """
    Args:
        data: 1D LongTensor of size: (n_tokens) (e.g., 5,543,556)
        bsz: batch size

    Returns:
        data: 2D LongTensor of size: (n_tokens/bsz, bsz) (e.g., 69294x80). This
            is just data reshaped into bsz columns. It is ordered down the 0th
            column, then down the 1st column, ..., until finally it reaches the
            bottom of the bsz'th column.
    """
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()

    # NOTE(mbforbes): Idea, but not doing right now: For BIG DATA, only move to
    # GPU in get_batch(...).
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    """
    Returns:
        data: 2D LongTensor of size (seq_len, bsz), which is the first seq_len
            rows of the full (batch matrix) dataset returned by batchify(...).

        target: 1D LongTensor of size (seq_len*bsz). This appears to be
        row-wise, the i+1'th row for each of the i rows in data, but without
        any shape.
    """
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))

    # NOTE(mbforbes): Idea, but not doing right now: Move to GPU here (instead
    # of in batchify).
    # if args.cuda:
    #     data = data.cuda()
    #     target = target.cuda()

    return data, target


class StreamingCorpus():
    """
    Replacement for data.Corpus.

    Requirements: must have fields:
        - train (can be used in batchify to produce StreamingBatchHolder)
        - valid (can be used in batchify to produce StreamingBatchHolder)
        - test (can be used in batchify to produce StreamingBatchHolder)
        - dictionary (likely just whatever the normal corpus does)
    """
    pass


class StreamingBatchHolder():
    """
    Replacement for result returned by batchify(...).

    Requirements: must support:
        - act as `source` in get_batch(...) called on it (could modify
        get_batch(...) above so it knows `source` might be this object)

        - size(...)

        - __len__
    """
    pass
