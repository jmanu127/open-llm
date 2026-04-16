import ast
import json
import regex as re
import numpy as np
from typing import Iterable, Iterator
import time

# compile once
PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
    re.UNICODE
)


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.bytes_to_id = {v: k for k, v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens_set = set(self.special_tokens)

        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self.special_split_re = re.compile("(" + "|".join(map(re.escape, ordered)) + ")")
        else:
            self.special_split_re = None

        self.cache = {}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(merges_filepath) as f:
            merges = [ast.literal_eval(line) for line in f if line.strip()]

        with open(vocab_filepath) as f:
            vocab_json = json.load(f)

        vocab = {int(k): ast.literal_eval(v) for k, v in vocab_json.items()}
        return cls(vocab, merges, special_tokens)

    def _split_special_tokens(self, text):
        if self.special_split_re:
            return self.special_split_re.split(text)
        return [text]

    def _get_pairs(self, word):
        return {(word[i], word[i + 1]) for i in range(len(word) - 1)}

    def _bpe(self, token: bytes):
        if token in self.cache:
            return self.cache[token]

        word = tuple(bytes([b]) for b in token)
        pairs = self._get_pairs(word)
        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
            if bigram not in self.merge_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            if len(word) == 1:
                break

            pairs = self._get_pairs(word)

        self.cache[token] = word
        return word

    def encode(self, text):
        tokens = []
        for chunk in self._split_special_tokens(text):
            if chunk in self.special_tokens_set:
                tokens.append(self.bytes_to_id[chunk.encode("utf-8")])
                continue
            for match in PAT.finditer(chunk):
                pre_token = match.group(0).encode("utf-8")
                merged = self._bpe(pre_token)
                for b in merged:
                    tokens.append(self.bytes_to_id[b])
        return tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token in self.encode(text):
                yield token

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def encode_dataset(tokenizer, input_path, output_path, dtype=np.uint16):
    print("Encoding:", input_path)

    all_tokens = []
    with open(input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            all_tokens.append(token_id)

    # Convert to NumPy array with explicit dtype (uint16)
    arr = np.array(all_tokens, dtype=dtype)

    # Save as a clean .npy file (no pickle)
    np.save(output_path, arr)

    print("Total tokens:", len(arr))


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "data/vocab.json",
        "data/merges.txt",
        special_tokens=["<|endoftext|>"]
    )

    start = time.time()

    encode_dataset(
        tokenizer,
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/train_tokens.npy"
    )

    encode_dataset(
        tokenizer,
        "data/TinyStoriesV2-GPT4-valid.txt",
        "data/val_tokens.npy"
    )

    print("Total time:", time.time() - start)