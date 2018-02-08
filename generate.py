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
from cache import VanillaLM, CacheLM
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
    parser.add_argument('--model', type=str, help='options: [vanilla,cache], default: cache', default='vanilla')
    parser.add_argument('--beam-size', type=int, default=5, help='beam size, default: 5')
    parser.add_argument('--sents', type=int, default=5, help='number of sentences to generate, default: 5')
    parser.add_argument('--use-eog', action='store_true', help='whether to add <end> as possible end of generation token (disabled by default)')
    parser.add_argument('--allow-unk', action='store_true', help='allow generating <unk> during beam search (disallowed by default)')
    parser.add_argument('--max-len', type=int, default=500, help='maximum generation length, default: 500')
    parser.add_argument('--theta', type=float, default=0.6625523432485668, help='theta controls cache flatness')
    parser.add_argument('--lmb', type=float, default=0.12785920428335693, help='lmb (lambda) controls mixture between cache (1.0) and LM `model` (0.0)')
    args = parser.parse_args()

    # TODO: just make model actual options
    model_opts = ['vanilla', 'cache']
    if args.model not in model_opts:
        print('ERROR: --model must be one of: {}'.format(', '.join(model_opts)))
        sys.exit(1)

    # dispaly settings
    print('INFO: Arguments:')
    for arg in vars(args):
        print('Info:   {:<14}: {}'.format(arg, getattr(args, arg)))

    return args


def tensor2str(vocab: data.Vocab, t: torch.LongTensor) -> str:
    return ' '.join(vocab.idx2word[idx] for idx in t)


def main():
    args = get_args()

    # load model and vocab
    print('INFO: Loading model and vocab')
    base_model = torch.load(args.model_path)
    base_model.cuda()  # TODO: probably a no-op. Verify this.
    base_model.eval()
    vocab = data.Vocab.load(args.vocab_path)

    # wrap underlying model in custom decoder
    if args.model == 'vanilla':
        model = VanillaLM(base_model)
    elif args.model == 'cache':
        model = CacheLM(base_model, len(vocab), args.theta, args.lmb)
    else:
        print('ERROR: Unknown model "{}"; aborting...'.format(args.model))
        sys.exit(1)

    # vocab indices.
    # TODO: old note: Note that we provide '<end>' as EOS, because we generate
    # multiple sentences, and `<end>` is how we indicate the end of a
    # generation.
    unk = vocab.word2idx[data.UNK]
    eos = vocab.word2idx['</s>']
    eog_set = {eos}
    special_tkns = {eos}
    eog_tkn = None
    unk_tkn = None
    if args.use_eog:
        eog_tkn = vocab.word2idx['<end>']
        eog_set.add(eog_tkn)
        special_tkns.add(eog_tkn)
    if not args.allow_unk:
        unk_tkn = vocab.word2idx['<unk>']
        special_tkns.add(unk_tkn)
    # beam_complete = beam.beam_complete_simple(eos)
    beam_complete = beam.beam_complete_nsents(args.sents, eos, eog_set)

    # load initials (as word idxes)
    print('INFO: Loading initials from "{}"'.format(args.initial_path))
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
    print('INFO: Generating and writing to "{}"'.format(args.output_path))
    with open(args.output_path, 'w') as f:
        for i, initial in enumerate(tqdm(initials)):
            # print('INFO: Initial {}/{}: {}'.format(
            #     i+1, len(initials), tensor2str(vocab, initial)))
            # create hidden state, which will just be zero'd out. batch size = 1.
            hidden = base_model.init_hidden(1)

            # clear any caching the model may have from previous initials
            model.clear()

            # feed in each token from the line. this provides context.
            for tkn in initial:
                inp.data.fill_(tkn)
                output, hidden = model.context(inp, hidden)

            # now, we can run w/ beam search.
            gen_tensor = beam.beamsearch(
                model, output, hidden, special_tkns, beam_complete,
                args.beam_size, args.max_len, eog_tkn, unk_tkn)
            gen_str = tensor2str(vocab, gen_tensor)

            # print('INFO: Generation: {}'.format(gen_str))
            f.write(gen_str)
            f.write('\n')


if __name__ == '__main__':
    main()
