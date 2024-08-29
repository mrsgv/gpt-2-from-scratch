"""
FineWeb-Edu dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
Downloads and tokenizes the data and saves data shards to disk.
Run simply as:
$ python fineweb.py
Will save shards to the local directory "edu_fineweb10B".
"""

import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp

# Configuration
local_dir = "edu_fineweb10B" # Local directory to save the tokenized data shards
remote_name = "sample-10BT" # Name of the dataset on the Hugging Face hub
shard_size = int(1e8)  # 100M tokens per shard

# Create the cache directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# This function will be called by each worker process to initialize the tokenizer
# make the tokenizer and special token global variables 
def init_tokenizer():
    global enc, eot
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']  # End of text token

# Tokenize Function
def tokenize(doc):
    """Tokenizes a single document and returns a numpy array of uint16 tokens"""
    global enc, eot
    # Tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]  # The special token delimits all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # GPT2 tokenizer has a vocabulary of 50257 tokens, so we can use uint16 to store them efficiently
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    """save numpy array to file"""
    np.save(filename, tokens_np)

if __name__ == '__main__':
    # Tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    
    # Use at least 1 CPU or half the available CPUs for tokenization
    nprocs = max(1, os.cpu_count() // 2)
    
    # Download the dataset
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    
    # Inspect the first document in the dataset
    # for key in fw[0].keys():
    #     print(key +" : "+ fw[0][key])
    #     print("\n -------------------------------------------------------- \n")
    # exit(0)

    # Create a pool of workers processes to tokenize the documents
    with mp.Pool(nprocs, initializer=init_tokenizer) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # Tokenize all documents and write output shards
        for tokens in tqdm(pool.imap(tokenize, fw, chunksize=16), desc="Processing documents"):
            # chuncksize tells pool.imap to split fw into chunks of 16 documents
            # a "document" is an individual entry in the dataset
            n_tokens = len(tokens)
            # print(f"n_tokens: {n_tokens}")
            # pass
            # Is there enough space in the current shard for the new tokens?
            if token_count + n_tokens < shard_size:
                # Simply append tokens to current shard
                all_tokens_np[token_count:token_count + n_tokens] = tokens
                token_count += n_tokens
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(n_tokens)
            else:
                # Write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
                # Split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)

                # progress to the next shard
                shard_index += 1
                progress_bar = None
                token_count = n_tokens - remainder
                # Populate the next shard with the leftovers of the current doc
                all_tokens_np[0:token_count] = tokens[remainder:]
                

        # Write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])




"""import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) # 100M tokens per shard, 100 shards in total

#create the cache the local directory if it doesn't exist
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingFaceFW/fineweb-edu", name = remote_name, split='train')

#init the tokenizers
enc = tiktoken.get_encoding('gpt2')
eot = enc._special_tokens['<|endoftext|>']

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Tokenization error"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # writes a numpy array of uint16 tokens to a binary file
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

# tokenize all the documents and write output shards
nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)  
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.npy")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0: len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
        
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"fineweb_{split}_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])
        """
