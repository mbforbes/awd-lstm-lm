"""
Continuous cache implementation (rather than rehashing pointer.py).

author: mbforbes
"""

# builtins
import code
from typing import Any, List, Tuple, Union

# 3rd party
import torch
from torch.autograd import Variable
import torch.nn.functional as F


FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]


class Cache(object):
    """
    No maximum length.
    """

    def __init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        """Clears cache."""
        self.hs = []  # type: List[Any]
        self.xs = []  # type: List[int]

    def add(self, h_prev: FloatTensor, x_cur: int) -> None:
        """
        Args:
            h_prev: previous hidden state (whatever x_cur would have come after)
            x_cur: token that came after h
        """
        self.hs.append(h_prev)
        self.xs.append(x_cur)

    def distribution(self, h: FloatTensor, vocab_size: int, theta: float) -> FloatTensor:
        """
        Returns probability distribution (not log!) over vocabulary according
        to cache, given h is the current observed hidden state.
        """
        # because size of cache is probably << vocab_size, construct by
        # iterating over that.
        dist = torch.cuda.FloatTensor(vocab_size).zero_()
        for i in range(len(self.hs)):
            ch = self.hs[i]
            x = self.xs[i]
            dist[x] += theta * ch.dot(h)
        return F.softmax(dist).data


class VanillaLM(object):
    """
    Wrapps a vanilla LM but provides the same API as the CacheLM.

    Don't be confused that this is in the cache.py file---it has no caching.
    It's just next to CacheLM for convenience of writing them with the same
    API. That way they can be swapped out for eachother.
    """

    def __init__(self, model) -> None:
        self.model = model

    def clear(self) -> None:
        """Clears any state. (No-op for VanillaLM.)"""
        pass

    def _forward(self, inp: Variable, hidden: Any) -> Tuple[FloatTensor, Any]:
        """
        Returns 2-tuple of (
            output -- a FloatTensor (not Variable) log-softmax distribution
                over the vocabulary,
            next_hidden --- whatever the hidden state is for the model,
        )
        """
        output, next_hidden = self.model(inp, hidden)
        return F.log_softmax(output.squeeze()).data, next_hidden

    def context(self, inp: Variable, hidden: Any) -> Tuple[FloatTensor, Any]:
        """
        context(...) indicates the model is being run forward through a context
        sentence. (For the VanillaLM, this isn't a distinction, it's just for
        API compatibility.)

        Returns 2-tuple of (
            output -- a FloatTensor (not Variable) log-softmax distribution
                over the vocabulary,
            next_hidden --- whatever the hidden state is for the model,
        )
        """
        return self._forward(inp, hidden)

    def predict(self, inp: Variable, hidden: Any) -> Tuple[FloatTensor, Any]:
        """
        predict(...) indicates the model is being run forward on its own
        without any ground truth labels (e.g., during generation). (Again, no
        difference for VanillaLM.)

        Returns 2-tuple of (
            output -- a FloatTensor (not Variable) log-softmax distribution
                over the vocabulary,
            next_hidden --- whatever the hidden state is for the model,
        )
        """
        return self._forward(inp, hidden)


class CacheLM(object):
    """Wraps a LM and uses a continuous Cache."""

    def __init__(
            self, model, vocab_size: int, theta: float, lmb: float) -> None:
        """
        - theta controls cache flatness
        - lmb controls mixture between cache (1.0) and LM `model` (0.0)
        """
        # save
        self.model = model
        self.vocab_size = vocab_size
        self.theta = theta
        self.lmb = lmb

        # init
        self.cache = Cache()
        self.h_prev = None
        self.h_prev_predict = None

    def clear(self) -> None:
        self.cache.clear()
        self.h_prev = None
        self.h_prev_predict = None

    def context(self, inp: Variable, hidden: Any) -> Tuple[FloatTensor, Any]:
        """
        context(...) indicates the model is being run forward through a context
        sentence. For the CacheLM, this builds the cache, but doesn't use it.

        inp should be a Variable containing a 1 x 1 LongTensor.

        Returns 2-tuple of (
            output -- a FloatTensor (not Variable) log-softmax distribution
                over the vocabulary,
            next_hidden --- whatever the hidden state is for the model,
        )
        """
        # update cache if we've seen at least one hidden state
        if self.h_prev is not None:
            self.cache.add(self.h_prev, inp.data[0][0])

        # run through model, getting rnn hidden states
        output, next_hidden = self.model(inp, hidden)

        # smush rnn hidden states into format we can work with
        # NOTE: this is equivalent to getting rnn_outputs from the underlying
        # model with return_h=True and then grabbing rnn_outputs[-1].
        self.h_prev = next_hidden[-1][0].data.squeeze()

        # return what's desired to caller
        return F.log_softmax(output.squeeze()).data, next_hidden

    def predict(self, inp: Variable, hidden: Any) -> Tuple[FloatTensor, Any]:
        """
        predict(...) indicates the model is being run forward on its own
        without any ground truth labels (e.g., during generation). For the
        CacheLM, the cache *is* used and *not* updated.

        inp should be a Variable containing a 1 x 1 LongTensor.

        Returns 2-tuple of (
            output -- a FloatTensor (not Variable) log-softmax
                distribution over the vocabulary,
            next_hidden --- whatever the hidden state is for the model,
        )
        """
        # get cache distribution
        p_cache = self.cache.distribution(
            hidden[-1][0].data.squeeze(), self.vocab_size, self.theta)

        # get LM distribution
        output, next_hidden = self.model(inp, hidden)
        p_lm = F.softmax(output.squeeze()).data

        # mix the two
        p_mixed = (1 - self.lmb) * p_lm + self.lmb * p_cache

        # return log prob dist and next hidden
        return p_mixed.log(), next_hidden
