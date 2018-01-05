"""
Beam search.

author: mbforbes
"""


# sketch:
#
# algorithm: sample
# - input: N: samples to draw
# - input: P: log probability distribution over words
# - get N indices based on setting: stochastic sample or N-argmax sample
# - return (indices, P[indices])  (the indices and their log probs)
#
# algorithm: stochastic sample
# - input: N: samples to draw
# - input: P: log probability distribution over words
# - # TODO: check multinomial sampling OK w/ log prob, or use exp()
# - return indices of multinomial sample N options from P
#
# algorithm: N-argmax sample
# - input: N: samples to draw
# - input: P: probability distribution over words
# - return indices of highest N values from P
#
# data structure: beam: [(
#       [word list],
#       hidden state (from last word),
#       sum of log probs so far
#   ),
#   ...
# ]
#
# algorithm: beam search
# - input: N: beam size
# - input: model with hidden state h initialized as desired (e.g., from context)
# - input: output from model (decoder) last token seen
# - init: beam = [([w], h, p(w)) for w, p(w) in sample(output, N)]
# - best_eos = ([EOS], p(EOS) in output)
# - finished = False
# - while not finished:
#   - next_beam = []
#   - for (words, h, prob_sum) in beam:
#       - # get prob dist
#       - # NOTE: pointer would be applied here to modify output (p(w))
#       - output, next_h = model(h)
#
#       - # consider EOS ending
#       - eos_prob = prob_sum + (p(EOS) in output)
#       - if eos_prob > best_eos[1]:
#           - best_eos = (words + [EOS], eos_prob)
#
#       - # grow next beam candidates
#       - # NOTE: using N+1 to account for getting EOS tokens (removed after)
#       - for (idx, p(idx)) in sample(output, N+1):
#           - next_beam.append((words + [idx], next_h, prob_sum + p(idx)))
#
#   - # prune next_beam and set to beam
#   - filter out EOS entries in next_beam
#   - beam = top-N highest prob_sum in next_beam
#
#   - # check stopping condition: when best_eos better than anything else in
#   - # beam.
#   finished = best_eos[1] > entry[2] for all entry in beam
