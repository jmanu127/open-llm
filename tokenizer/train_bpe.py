import json
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from typing import BinaryIO

import regex as re

# compile once at module load
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", re.UNICODE)
special_tokens = ["<|endoftext|>"]
split_re = re.compile("(" + "|".join(map(re.escape, special_tokens)) + ")")


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(start: int, end: int, input_path: str, special_tokens: list[str]) -> dict:
    """
    pretokenize the data for gpt2 regex
    """
    freq = defaultdict(int)

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # moved outside of function to compile once since special token isn't changing
    # split_re = re.compile("(" + "|".join(map(re.escape, special_tokens)) + ")")
    text_splits = split_re.split(chunk)

    for text in text_splits:
        if text in special_tokens:
            freq[(text.encode("utf-8"),)] += 1
            continue
        for m in PAT.finditer(text):
            b = m.group().encode("utf-8")
            # faster tuple construction
            s = tuple(b[i : i + 1] for i in range(len(b)))
            freq[s] += 1

    return freq


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a byte-level BPE tokenizer on the data in input_path.

    Args:
        input_path: str Path to a text file with BPE tokenizer training data.
        vocab_size: int A positive integer that defines the maximum final vocabulary size (including the
            initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: list[str] A list of strings to add to the vocabulary. These special tokens do not
            otherwise affect BPE training.

    Returns:
        vocab: dict[int, bytes] The tokenizer vocabulary, a mapping from int (token ID in the vocabu-
            lary) to bytes (token bytes).
        merges: list[tuple[bytes, bytes]] A list of BPE merges produced from training. Each list item
            is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
            <token2>. The merges should be ordered by order of creation.
    """
    # vocab init
    vocab = {i: bytes([i]) for i in range(256)}
    offset = len(vocab)

    vocab.update({i + offset: tok.encode("utf-8") for i, tok in enumerate(special_tokens)})

    len_vocab = len(vocab)
    # print(f"\nInitial vocabulary size: {len_vocab}")

    # pre-tokenization
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            b"<|endoftext|>",  # encode once
        )

    freq = defaultdict(int)
    # freq = Counter()

    with multiprocessing.Pool(num_processes) as pool:
        for freq_chunk in pool.starmap(
            pretokenize_chunk,
            ((start, end, input_path, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])),
        ):
            for s, c in freq_chunk.items():
                freq[s] += c

    # print("Pre-tokenization complete.")

    # count frequency
    num_byte_pairs = defaultdict(int)
    loc_byte_pairs = defaultdict(dict)

    for s, s_freq in freq.items():
        n = len(s)
        if n < 2:
            continue

        # collect locations of pairs per string
        byte_pair_to_index = defaultdict(list)
        for j in range(n - 1):
            byte_pair_to_index[(s[j], s[j + 1])].append(j)

        # update global structures
        for byte_pair, idxs in byte_pair_to_index.items():
            count = s_freq * len(idxs)
            num_byte_pairs[byte_pair] += count  # number of byte pairs
            loc_byte_pairs[byte_pair][s] = idxs  # byte pair, token, index of token
    # print("Count frequency of pairs complete.")

    # merge tokens
    merges = []

    while len_vocab < vocab_size:
        # find best pair
        merged_pair = max(num_byte_pairs.items(), key=lambda x: (x[1], x[0]))[0]

        merges.append(merged_pair)
        vocab[len_vocab] = merged_pair[0] + merged_pair[1]
        len_vocab += 1

        # iterate directly
        affected_strings = list(loc_byte_pairs[merged_pair])
        for s in affected_strings:
            s_freq = freq[s]

            # merge string
            s_new = []
            j = 0
            n = len(s)

            while j < n - 1:
                if s[j] == merged_pair[0] and s[j + 1] == merged_pair[1]:
                    s_new.append(s[j] + s[j + 1])
                    j += 2
                else:
                    s_new.append(s[j])
                    j += 1

            if j == n - 1:
                s_new.append(s[j])

            s_new = tuple(s_new)

            # accumulate frequency
            # freq[s_new] = freq.get(s_new, 0) + s_freq
            freq[s_new] += s_freq

            # remove old pair contributions
            for j in range(n - 1):
                pair = (s[j], s[j + 1])
                num_byte_pairs[pair] -= s_freq
                loc_byte_pairs[pair].pop(s, None)

            # add new pair contributions
            for j in range(len(s_new) - 1):
                pair = (s_new[j], s_new[j + 1])
                num_byte_pairs[pair] += s_freq
                loc_byte_pairs[pair][s_new] = loc_byte_pairs[pair].get(s_new, []) + [j]

            del freq[s]

        # fully remove merged pair
        del num_byte_pairs[merged_pair]
        del loc_byte_pairs[merged_pair]

    # print(f"Merge complete. Final vocab size: {len_vocab}")
    return vocab, merges


if __name__ == "__main__":
    input_path = "/data/TinyStoriesV2-GPT4-train.txt"

    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    start = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens, num_processes=10)
    end = time.time()

    print("Total time taken", end - start)

    result_path = "/data/"
    with open(f"{result_path}merges.txt", "w") as f:
        for line in merges:
            f.write(f"{line}\n")
    with open(f"{result_path}vocab.json", "w") as file:
        json.dump({k: str(vocab[k]) for k in vocab}, file)

    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    print("Longest token", vocab[longest_token_id])
