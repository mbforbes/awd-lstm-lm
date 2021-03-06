"""
Beam search.

author: mbforbes
"""

# builtins
import code
from typing import Union, Tuple, List, Callable, Set, Optional, Any, NamedTuple

# third party
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# local
import data


LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]
BeamCompleteFn = Callable[[List[int]], bool]

class BeamEntry(NamedTuple):
    tokens: List[int]
    hidden: Any
    logprob: float

class BestComplete(NamedTuple):
    tokens: List[int]
    logprob: float

# TODO: stochastic sampling (alternative to 'topk'). This algorithm looks like:
# - input: P: log probability distribution over words
# - input: N: samples to draw
# - NOTE: unsure of can use torch.multinomial because indices not returned...
# - return values, indices of multinomial sample N options from P

def beam_complete_simple(eos: int) -> BeamCompleteFn:
    """
    Returns a function that checks whether a beam (words only) ends in an EOS.
    """
    def ends_in_eos(all_words: List[int]) -> bool:
        return all_words[-1] == eos
    return ends_in_eos


def beam_complete_nsents(n: int, eos: int, eog: Set[int]) -> BeamCompleteFn:
    """
    Returns a function that checks whether a beam (words only) contains `n`
    `eos` tokens before its final token, and then ends in one of the tokens in
    `eog`.
    """
    def nsents_then_eog(all_words: List[int]) -> bool:
        n_eos = len([tkn for tkn in all_words[:-1] if tkn == eos])
        return n_eos == n - 1 and all_words[-1] in eog
    return nsents_then_eog


def beamsearch(
        model, output, hidden, special_tkns: Set[int],
        beam_complete: BeamCompleteFn, beam_size: int = 5,
        maxlen: int = 500, eog: Optional[int] = None,
        unk: Optional[int] = None) -> torch.LongTensor:
    """
    Model is a VanillaLM or CacheLM.

    Args:
        special_tkns: these are tokens (like EOS, EOG, UNK) that may or may not
            appear in the beam, but don't always count as normal words. This
            means that when we grab candidates, we grab this many more than our
            beam size to account for getting them.

        eog: if an int is provided, will disallow this token from appearing at
            a non-final location in a set of words. if None, ignored.

        unk: if an int is provided, will disallow this token from appearing
            anywhere in a set of words. if None, ignored.

    Each entry in the beam is a 3-tuple of (
        [words]: List[int] (sequence of tokens),
        hidden: any (after running all EXCEPT last word through model),
        log_prob: float (log probability score of entire sequence so far),
    )
    """
    # init beam.
    # ---
    lsm_output = F.log_softmax(output.squeeze()).data

    # grab beam_size + 1 because we may grab special tokens
    values: List[float]
    indices: List[int]
    values, indices = lsm_output.topk(beam_size + len(special_tkns))
    beam: List[BeamEntry] = []

    for i in range(len(values)):
        # don't add any special tokens
        if indices[i] in special_tkns:
            continue
        beam.append(BeamEntry([indices[i]], hidden, values[i]))

    # if no special tokens were added, need to prune back down to beam size
    if len(beam) > beam_size:
        beam = beam[:beam_size]

    # init the best complete entry (to invalid)
    # ---
    best_complete = BestComplete([-1], float('-inf'))

    # beam search
    # ---
    inp = Variable(torch.cuda.LongTensor(1,1), volatile=True)
    finished = False
    while (not finished) and len(beam[0].tokens) < maxlen:
        next_beam: List[BeamEntry] = []
        for (words, hidden, prob_sum) in beam:
            # run the last beam word through model
            inp.data.fill_(words[-1])
            output, next_hidden = model.predict(inp, hidden)
            lsm_output = F.log_softmax(output.squeeze()).data

            # grow next beam candidates
            values, indices = lsm_output.topk(beam_size + len(special_tkns))
            for i in range(len(values)):
                # maybe prevent unks
                if unk is not None and indices[i] == unk:
                    continue
                # candidate computations
                new_logprob = prob_sum + values[i]
                new_words = words + [indices[i]]
                if beam_complete(new_words):
                    if new_logprob > best_complete.logprob:
                        best_complete = BestComplete(new_words, new_logprob)
                # maybe prevent eog token for incomplete beam
                elif eog is not None and indices[i] == eog:
                    continue
                else:
                    next_beam.append(BeamEntry(new_words, next_hidden, new_logprob))

        # prune to top beam_size entries by logprob to make next beam.
        beam = sorted(next_beam, key=lambda entry: entry.logprob, reverse=True)[:beam_size]

        # check finishing condition. (beam sorted right now so first has
        # highest score.)
        finished = best_complete.logprob > beam[0].logprob

    # If we've got a best_complete that's better than the whole beam we return
    # it. If instead we hit the max length, we still return the best complete
    # entry, because we want it to end.
    return torch.LongTensor(best_complete.tokens)
