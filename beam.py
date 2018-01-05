"""
Beam search.

author: mbforbes
"""

# builtins
import code
from typing import Union, Tuple

# third party
import torch
from torch.autograd import Variable
import torch.nn.functional as F

# local
import data


LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]


# TODO: stochastic sampling (alternative to 'topk'). This algorithm looks like:
# - input: P: log probability distribution over words
# - input: N: samples to draw
# - NOTE: unsure of can use torch.multinomial because indices not returned...
# - return values, indices of multinomial sample N options from P


def beamsearch(
        model, output, hidden, eos: int, beam_size: int = 5,
        maxlen: int = 500) -> torch.LongTensor:
    """
    Each entry in the beam is a 3-tuple of (
        [words]: List[int] (sequence of tokens),
        hidden: any (after running all EXCEPT last word through model),
        log_prob: float (log probability score of entire sequence so far),
    )
    """
    # init beam.
    # ---
    lsm_output = F.log_softmax(output.squeeze()).data

    # grab beam_size + 1 because we may grab EOS
    values, indices = lsm_output.topk(beam_size + 1)
    beam = []

    for i in range(len(values)):
        # don't add EOS
        if indices[i] == eos:
            continue
        beam.append(([indices[i]], hidden, values[i]))

    # if no EOS was added, need to prune down one
    if len(beam) > beam_size:
        beam = beam[:-1]

    # init the best eos entry
    # ---
    best_eos = ([eos], lsm_output[eos])

    # beam search
    # ---
    inp = Variable(torch.cuda.LongTensor(1,1), volatile=True)
    finished = False
    while (not finished) and len(beam[0]) < maxlen:
        next_beam = []
        for (words, hidden, prob_sum) in beam:
            # NOTE: pointer would modify output here.
            # run the last beam word through model
            inp.data.fill_(words[-1])
            output, next_hidden = model(inp, hidden)
            lsm_output = F.log_softmax(output.squeeze()).data

            # consider EOS ending
            eos_prob = prob_sum + lsm_output[eos]
            if eos_prob > best_eos[1]:
                best_eos = (words + [eos], eos_prob)

            # grow next beam candidates
            # NOTE: using N+1 to account for getting EOS tokens
            values, indices = lsm_output.topk(beam_size + 1)
            for i in range(len(values)):
                if indices[i] == eos:
                    continue
                next_beam.append((words + [indices[i]], next_hidden, prob_sum + values[i]))

        # prune to make next beam.
        beam = sorted(next_beam, key=lambda entry: entry[2], reverse=True)[:beam_size]

        # check finishing condition. (beam sorted right now so first has
        # highest score.)
        finished = best_eos[1] > beam[0][2]

    # If we've got a best_eos that's better than the whole beam we return it.
    # If instead we hit the max length, we still return the best EOS entry,
    # because we want it to end.
    return torch.LongTensor(best_eos[0])
