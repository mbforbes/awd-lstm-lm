###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse
import code

from typing import List
import torch
from torch.autograd import Variable

import data
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--initial', type=str, default='./data/rocstory/rocstory_test_initials.txt',
                    help='path of the data to initialize the LM from')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

# build stuff once
corpus = data.Corpus(args.data, False)
ntokens = len(corpus.dictionary)

# build up initial we'll use
with open(args.initial, 'r') as f:
    initials = [line.strip().split(' ') for line in f.readlines()]

# old: init hidden state randomly
# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
# if args.cuda:
#     input.data = input.data.cuda()

# new: basically the same; just make some hidden state which we will
# immediately fill with the initial
input = Variable(torch.cuda.LongTensor(1,1), volatile=True)
gen_lines = []  # type: List[str]

# loop over all initials to generate endings for each line
for initial in tqdm(initials):
    # zero out hidden state. batch size = 1. OK.
    hidden = model.init_hidden(1)

    # feed in all tokens from the line
    for tkn in initial:
        # i think they don't handle unks because they build their vocabulary
        # from the test set. oops. i guess we'll just use index 0 (looks like
        # it's '<beg>') as unk.
        idx = corpus.dictionary.word2idx.get(tkn)
        if idx is None:
            idx = 0
        input.data.fill_(idx)

        # run through model
        output, hidden = model(input, hidden)

    # now start actually generating
    gen_words = []

    # we'll hard cutoff at args.words, but really, we'll stop much before
    # (probably) whenver we hit the end of a sentence.
    for i in range(args.words):
        # forward, then sample distribution to pick word
        output, hidden = model(input, hidden)
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]

        # fill in word as next hidden state
        input.data.fill_(word_idx)

        # get actual word
        word = corpus.dictionary.idx2word[word_idx]

        # break if we're able to stop
        if word in ['</s>', '<end>', '<eos>']:
            break

        # save as output
        gen_words.append(word)


    # combine words into line and save to output
    gen_lines.append(' '.join(gen_words))

# report results!
with open(args.outf, 'w') as outf:
    for line in gen_lines:
        outf.write(line + '\n')
