from pretokenization_example import find_chunk_boundaries
import multiprocessing
from pathlib import Path
import re

class BPETokenizer():
    
    def __init__(self):
        self.PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def _pre_tokenize(self, corpus_chunk: str, freq_tables: list[dict[tuple[bytes]: int]]):
        freq = {}
        for pre_token in re.finditer(self.PAT, corpus_chunk):
            freq[tuple(pre_token.group().encode("utf-8"))] += 1
        freq_tables.append(freq)
    

    def train(self, corpus: Path, vocab_size: int, special_tokens: list[str]):
        with open(corpus, "rb") as f:
            num_workers = 16
            boundaries = find_chunk_boundaries(f, num_workers, b"<|endoftext|>")

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                

