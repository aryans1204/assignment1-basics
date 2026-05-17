import os
from typing import BinaryIO, Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path
import regex as re
import bpe_tokenizer
from collections import Counter
import pickle


PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
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

def _pre_tokenize(corpus_with_tokens: tuple[str, list[str]], encoding=False) -> dict[str, int] | list[str]:
    freq = {}
    corpus_chunk, special_tokens = corpus_with_tokens
    if encoding:
        # while encoding, keep the special tokens
        return re.finditer(PAT, corpus)

    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = f"{"|".join(escaped_tokens)}"
    delimited_corpus = re.split(pattern, corpus_chunk)
    for corpus in delimited_corpus:
        for pre_token in re.finditer(PAT, corpus):
            encoded_tup = pre_token.group()
            if encoded_tup not in freq:
                freq[encoded_tup] = 0

            freq[encoded_tup] += 1
    return freq

class BPETokenizer():
    
    def __init__(self, vocab: dict[tuple[int, bytes]], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.inv_vocab = {val: key for (key, val) in vocab}
        self.merges = merges
        self.special_tokens = special_tokens
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
            Construct Tokenizer from a vocab and merges filepath
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
            with open(merges_filepath, "rb") as merge:
                merges = pickle.load(merge)
                return cls(vocab, merges, special_tokens)
            
    def encode(self, text: str) -> list[int]:
        """
            Encode a given string into a list of tokens
        """
        tokens = []
        pre_tokens = _pre_tokenize((text, self.special_tokens), encoding=True)
        for pre_token in pre_tokens:
            if pre_token in self.special_tokens:
                tokens.append(self.inv_vocab[pre_token.encode("utf-8")])
                continue
            
            merged_idx = 0
            merged_pairs = set([pre_token[i] for i in range(len(pre_token)-1)])
            indices = {pre_token[i]: i for i in range(len(pre_token))}

            while pre_token not in merged_pairs and merged_idx < len(self.merges):
                merge_1, merge_2 = self.merges[merged_idx]
                if merge_1 in merged_pairs and merge_2 in merged_pairs and indices[merge_1]+len(merge_1) == indices[merge_2]:
                    merged_pair = merge_1+merge_2
                    tokens.append(self.inv_vocab[merged_pair.encode("utf-8")])
                    merged_pairs.remove(merge_1)
                    merged_pairs.remove(merge_2)
                    merged_pairs.add(merged_pair)
                    indices[merged_pair] = indices[merge_1]
                if pre_token not in merged_pairs:
                    for m in merged_pairs:
                        if len(m) == 1:
                            tokens.append(self.inv_vocab[m.encode('utf-8')])
        return tokens
    
    def encode_itreable(self, text: Iterable) -> Iterator:
        for t in text:
            yield self.encode(text)
    
    def decode(self, token_ids: list[int]) -> str:
        decoded = bytearray()
        for token_id in token_ids:
            decoded.append(self.vocab[token_id])
        
        return decoded.decode("utf-8")
    
    @staticmethod
    def train(corpus: Path, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(corpus, "rb") as f:
            num_workers = 8
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

            chunks = []
        
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))
            
            with Pool(num_workers) as pool:
                # parallelize with n_workers processes and create Counters
                counters = [Counter(freq) for freq in pool.map(_pre_tokenize, [(chunk, special_tokens) for chunk in chunks])]
                freq_counter = counters[0]
                for counter in counters[1:]:
                    freq_counter += counter
                
                bytes_freq = dict(freq_counter)
                trained_tokenizer = bpe_tokenizer.train_bpe_tokenizer(bytes_freq, vocab_size-(256+len(special_tokens)), 255+len(special_tokens))
                token_ids = {i: bytes([i]) for i in bytes(range(256))} | {256+i: special_tokens[i].encode("utf-8") for i in range(len(special_tokens))} | {id: trained_tokenizer.token_ids[id] for id in range(256+len(special_tokens), len(trained_tokenizer.token_ids))}
                merges = [(merged_pair[0], merged_pair[1]) for merged_pair in trained_tokenizer.merged_pairs]
                return token_ids, merges
                
    
if __name__ == "__main__":
    BPETokenizer().train(corpus=Path("tests/fixtures/corpus.en"), vocab_size=10000, special_tokens=["<|endoftext|>"])

                

