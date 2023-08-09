#usr/bin/python
"""
embeddings.py
"""

import torch
import gzip
from multiprocessing import Pool, cpu_count

# Embeddings from pre-trained models
@torch.no_grad()
def forward_fn_pooler(model, inputs):
    return model(**inputs).pooler_output

@torch.no_grad()
def forward_fn_cls(model, inputs):
    return model(**inputs).last_hidden_state[:, 0, :]

def get_embeddings(sents, tokenizer, model, forward_fn, config):
    inputs = tokenizer(sents, 
              padding=True, 
              truncation=True, 
              max_length=config.max_length,
              return_tensors='pt')
    
    inputs = {k: v.to(config.device) for k, v in inputs.items()}
    return forward_fn(model, inputs)

def collate_samples(samples, tokenizer, config):
    """
    Convert a list of samples into a batch of tokenized sentences
    while retaining information about their sample origins.
    """
    all_sentences = [sentence for sample in samples for sentence in sample] # (n_sentences,)
    tokenized_sentences = tokenizer(all_sentences, 
                                    padding=True, 
                                    truncation=True, 
                                    max_length=config.max_length, 
                                    return_tensors='pt') # (n_sentences, max_length)

    inputs = {k: v.to(config.device) for k, v in tokenized_sentences.items()} # (n_sentences, max_length)
    sample_boundaries = [len(sample) for sample in samples] # (n_samples,)
    
    return inputs, sample_boundaries

def get_batched_embeddings(samples, tokenizer, model, forward_fn, config):
    inputs, sample_boundaries = collate_samples(samples, tokenizer, config)
    embeddings = forward_fn(model, inputs) # (n_sentences, embedding_size)
    
    # Split the embeddings according to the sample boundaries
    sample_embeddings = torch.split(embeddings, sample_boundaries) # (n_samples, n_sentences, embedding_size)

    return sample_embeddings


# Compression-based embeddings

def get_distance(s1, s2, cs1, cs2):
    lcs1 = len(cs1)
    lcs2 = len(cs2)
    lcss = len(get_compression(' '.join([s1, s2])))
    return (lcss - min(lcs1, lcs2)) / max(lcs1, lcs2)

def get_compression(sent):
    return gzip.compress(sent.encode('utf-8'))

def compute_single_sample_distance(sample):
    compressions = [get_compression(s) for s in sample]
    n = len(sample)
    distances_2d = [[get_distance(sample[i], sample[j], compressions[i], compressions[j]) for j in range(n)] for i in range(n)]
    return distances_2d

def compute_chunk_distances(samples_chunk):
    # For each sample in the chunk, compute distances
    return [compute_single_sample_distance(sample) for sample in samples_chunk]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_compression_distance(samples):
    n_cores = cpu_count()  # this gets the number of available CPU cores
    n_samples = len(samples)
    
    # Calculate chunk size based on the number of available cores
    chunk_size = n_samples // n_cores if n_cores != 0 else n_samples
    
    # Use a generator to create chunks
    samples_gen = chunks(samples, chunk_size)
    
    with Pool(processes=n_cores) as pool:
        results = pool.map(compute_chunk_distances, samples_gen)

    # Flatten the results back into a list
    results = [item for sublist in results for item in sublist]
    
    # Convert each result to a torch tensor
    tensors = [torch.tensor(distances_2d, dtype=torch.float) for distances_2d in results]

    return tensors
