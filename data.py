# builtins
import argparse
import code
from collections import Counter
import os
import pickle
from typing import Optional, Union
import typing

# 3rd party
from tqdm import tqdm
import torch


LongTensor = Union[torch.LongTensor, torch.cuda.LongTensor]


## ---
## Provided classes
## ---

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, test):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        if test:
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in tqdm(f):
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in tqdm(f):
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

## ---
## Custom replacements (different APIs)
## ---


class Vocab(object):

    # settings
    UNK = '<unk>'
    EOS = '</s>'
    SPECIAL_TOKENS = [UNK, EOS]

    def __init__(self):
        # state
        self.word2idx = {}
        self.idx2word = []

    @staticmethod
    def build(
            train_path: str, save_path: Optional[str] = None,
            limit: Optional[int] = None) -> 'Vocab':
        """
        Factory.

        Args:
            train_path
            save_path: where to save the object to. If None, doesn't save.
            limit: if None, no vocab limit. Else, max vocab size (excluding any
                special tokens this code adds).
        """
        print('INFO: Building vocabulary for "{}"'.format(train_path))

        # Read in all words in training set
        # TODO: nearly same thing used below; should bake into a generator.
        c = Counter()  # type: typing.Counter[str]
        with open(train_path, 'r') as f:
            for line in tqdm(f):
                words = line.split()
                # != '<end>' check specific to our datasets
                if len(words) > 0 and words[-1] != '<end>':
                    words.append(Vocab.EOS)
                c.update(words)

        # most common `limit` words
        if limit is not None:
            limit += len(Vocab.SPECIAL_TOKENS)
        v = Vocab()
        for i, (token, freq) in enumerate(c.most_common(limit)):
            v.idx2word.append(token)
            v.word2idx[token] = i

        # always add unk
        v.idx2word.append(Vocab.UNK)
        v.word2idx[Vocab.UNK] = len(v.idx2word) - 1

        print('INFO: Vocab size (incl. special tkns): {}'.format(len(v.idx2word)))

        # maybe save
        if save_path is not None:
            print('INFO: Saving vocabulary to "{}"'.format(save_path))
            v.save(save_path)

        return v

    @staticmethod
    def load(path: str) -> 'Vocab':
        """
        Factory. Loads something built from Vocab.build(...).
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    def save(self, path: str) -> None:
        """
        Pickles self to path.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def tokenize(self, path: str) -> torch.LongTensor:
        """
        Returns 1D tensor (CPU) with path tokenized using self.
        """
        # Also NOTE: For later, build convenience function (to be called
        # separately) to do preproessing: build (and save) vocab, tokenize (and
        # save) all files. Then, main code route can just load the pickles and
        # tensors.

        # Current approach: build up huge token array and then convert to
        # tensor in one go.
        print('INFO: Tokenizing "{}"'.format(path))

        unk = self.word2idx[Vocab.UNK]

        tkns = []
        with open(path, 'r') as f:
            for line in tqdm(f):
                # get tkns in line
                words = line.split()
                # != '<end>' check specific to our datasets
                if len(words) > 0 and words[-1] != '<end>':
                    words.append(Vocab.EOS)

                # convert to ints and save to our buffer. UNK if missing.
                for word in words:
                    tkns.append(self.word2idx.get(word, unk))

        return torch.LongTensor(tkns)

    @staticmethod
    def preprocess(
            in_train_path: str, in_val_path: Optional[str],
            in_test_path: Optional[str], in_vocab_limit: Optional[int],
            out_vocab_path: Optional[str], out_train_path: Optional[str],
            out_val_path: Optional[str], out_test_path: Optional[str]) -> None:
        """
        Convenience function.

        If vocab limit is None, no limit is applied. If vocab path not set,
        doesn't save it. If out train path is None, skips tokenizing and saving
        train. If either val path is None, skips tokenizing and saving val.
        If either test path is None, skips tokenizing and saving test.
        """
        v = Vocab.build(in_train_path, out_vocab_path, in_vocab_limit)

        worklist = [
            (in_train_path, out_train_path),
            (in_val_path, out_val_path),
            (in_test_path, out_test_path),
        ]

        for in_path, out_path in worklist:
            if in_path is None or out_path is None:
                continue
            torch.save(v.tokenize(in_path), out_path)


def main():
    """
    Just tokenizing here because it's convenient.
    """
    parser = argparse.ArgumentParser()
    # required
    parser.add_argument(
        'in_train_path',
        type=str,
        help='path to training file')
    # all the rest optional
    parser.add_argument(
        '--in_val_path',
        type=str,
        help='path to validation file')
    parser.add_argument(
        '--in_test_path',
        type=str,
        help='path to test file')
    parser.add_argument(
        '--in_vocab_limit',
        type=int,
        help='vocab limit')
    parser.add_argument(
        '--out_vocab_path',
        type=str,
        help='path to save Vocab object as pickle')
    parser.add_argument(
        '--out_train_path',
        type=str,
        help='path to save prcessed train file as tensor (of token idxes)')
    parser.add_argument(
        '--out_val_path',
        type=str,
        help='path to save prcessed val file as tensor (of token idxes)')
    parser.add_argument(
        '--out_test_path',
        type=str,
        help='path to save prcessed test file as tensor (of token idxes)')
    args = parser.parse_args()

    Vocab.preprocess(
        args.in_train_path,
        args.in_val_path,
        args.in_test_path,
        args.in_vocab_limit,
        args.out_vocab_path,
        args.out_train_path,
        args.out_val_path,
        args.out_test_path)


if __name__ == '__main__':
    main()
