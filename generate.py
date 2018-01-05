"""
Generates w/ context and beam search.

author: mbforbes
"""

# builtins
import argparse
import code
import sys

# 3rd party
import torch
from torch.autograd import Variable
from tqdm import tqdm

# local
import beam
import data
from data import Vocab


def get_args():
    """
    Also does generic setup.
    """
    # pre-check: ensure CUDA. can adapt this for non-cuda later if we need to;
    # for now, easier to avoid checks.
    if not torch.cuda.is_available():
        print('ERROR: Code assumes CUDA available.')
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='path to trained pytorch model to use')
    parser.add_argument('vocab_path', type=str, help='path to vocab (that the model used)')
    parser.add_argument('initial_path', type=str, help='path to file containing initials')
    parser.add_argument('output_path', type=str, help='path to file to write output generations')
    parser.add_argument('--beam-size', type=int, default=5, help='beam size')
    parser.add_argument('--max-len', type=int, default=500, help='maximum generation length')
    return parser.parse_args()


def tensor2str(vocab: data.Vocab, t: torch.LongTensor) -> str:
    return ' '.join(vocab.idx2word[idx] for idx in t)


def main():
    args = get_args()

    # load model and vocab

    model = torch.load(args.model_path)
    model.cuda()  # TODO: probably a no-op. Verify this.
    model.eval()
    vocab = data.Vocab.load(args.vocab_path)

    # vocab indices. Note that we provide '<end>' as EOS, because we generate
    # multiple sentences, and `<end>` is how we indicate the end of a
    # generation.
    unk = vocab.word2idx[data.UNK]
    eos = vocab.word2idx['<end>']

    # load initials (as word idxes)
    initials = []
    with open(args.initial_path, 'r') as f:
        for line in f.readlines():
            words = line.strip().split(' ')
            initials.append(
                torch.LongTensor([vocab.word2idx.get(w, unk) for w in words])
            )

    # sanity check
    # print(tensor2str(vocab, initials[0]))
    # code.interact(local=dict(globals(), **locals()))

    # prep input (we'll fill with each token as we go)
    inp = Variable(torch.cuda.LongTensor(1,1), volatile=True)

    # actual generation
    generations = []
    for i, initial in enumerate(initials):
        # print('INFO: Initial {}/{}: {}'.format(
        #     i+1, len(initials), tensor2str(vocab, initial)))
        # create hidden state, which will just be zero'd out. batch size = 1.
        hidden = model.init_hidden(1)

        # feed in each token from the line. this provides context.
        for tkn in initial:
            inp.data.fill_(tkn)
            output, hidden = model(inp, hidden)

        # now, we can run w/ beam search.
        gen_tensor = beam.beamsearch(
            model, output, hidden, eos, args.beam_size, args.max_len)
        gen_str = tensor2str(vocab, gen_tensor)
        generations.append(gen_str)
        print('INFO: Generation: {}'.format(gen_str))

    # write to file.
    print('INFO: Writing to "{}"'.format(args.output_path))
    with open(args.output_path, 'w') as f:
        for line in generations:
            f.write(line)
            f.write('\n')


if __name__ == '__main__':
    main()
