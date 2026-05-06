from pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from pathlib import Path
import regex as re
import bpe_tokenizer
from collections import Counter

PAT =  r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _pre_tokenize(corpus_with_tokens: tuple[str, list[str]]) -> dict[str: int]:
    freq = {}
    corpus_chunk, special_tokens = corpus_with_tokens
    pattern = f"{"|".join(special_tokens)}"
    delimited_corpus = re.split(pattern, re.escape(corpus_chunk))
    for corpus in delimited_corpus:
        for pre_token in re.finditer(PAT, corpus):
            encoded_tup = pre_token.group()
            if encoded_tup not in freq:
                freq[encoded_tup] = 0

            freq[encoded_tup] += 1
    return freq

class BPETokenizer():
    
    def __init__(self):
        pass
    
    @staticmethod
    def train(corpus: Path, vocab_size: int, special_tokens: list[str]):
        with open(corpus, "rb") as f:
            num_workers = 4
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
                trained_tokenizer = bpe_tokenizer.train_bpe_tokenizer(bytes_freq, vocab_size-(256+len(special_tokens)), 256+len(special_tokens))
                
    
if __name__ == "__main__":
    BPETokenizer().train(corpus=Path("data/TinyStoriesV2-GPT4-valid.txt"), vocab_size=10000, special_tokens=["<|endoftext|>"])

                

