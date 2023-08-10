#usr/bin/python
"""
scoring.py
"""

import torch
from torch.nn.functional import cosine_similarity as cos
from .utils import normalize_weights

def sentence_content_score(sents_emb):
    """
    Computes the content scores for sentences based on cosine similarity with the centroid.

    Args:
    - sents_emb (tensor): 
        Embeddings for sentences.

    Returns:
    - tensor: 
        Content scores for each sentence.
    """
    centroid = sents_emb.mean(dim=0)
    return cos(sents_emb, centroid, dim=1)

def sentence_novelty_score(sents_emb, content_scores, tau):
    """
    Computes the novelty scores for sentences by comparing their similarity with other sentences.

    Args:
    - sents_emb (tensor): 
        Sentence embeddings.
    - content_scores (tensor): 
        Previously computed content scores for the sentences.
    - tau (float): 
        Threshold for novelty score computation.

    Returns:
    - tensor: 
        Novelty scores for each sentence.
    """
    # Compute similarity matrix
    sim = cos(sents_emb[:,:,None], sents_emb.t()[None,:,:], dim=1) # pairwise cosine similarity
    sim = sim.fill_diagonal_(float('-inf'))
    
    # Get max values and its indices
    max_vals, max_idxs = sim.max(dim=1)
    
    # Compute novelty scores
    novelty_scores = torch.zeros(len(max_vals))
    for i in range(len(max_vals)):
        if max_vals[i] < tau:
            novelty_scores[i] = 1
        elif max_vals[i] > tau and content_scores[i] > content_scores[max_idxs[i]]:
            novelty_scores[i] = 1
        else:
            novelty_scores[i] = 1 - max_vals[i]

    return novelty_scores

def sentence_position_score(sents_emb):
    """
    Calculates position scores for sentences based on their order in the text.

    Args:
    - sents_emb (tensor): 
        Sentence embeddings.

    Returns:
    - tensor: 
        Position scores for each sentence.
    """
    M = sents_emb.shape[0]
    sentence_positions = torch.tensor([i for i in range(M)])
    scores = torch.max(torch.tensor(0.5), torch.exp(-sentence_positions / (3 * torch.sqrt(torch.tensor(M, dtype=torch.float)))))
    return scores

def sentence_scores(sents_emb, tau=0.95, alpha=0.6, beta=0.2, gamma=0.2, normalize_weights_fn=normalize_weights):
    """
    Computes the overall scores for sentences considering content, novelty, and position.

    Args:
    - sents_emb (tensor): 
        Sentence embeddings.
    - tau (float, optional): 
        Threshold for novelty score computation. Default is 0.5.
    - alpha (float, optional): 
        Weight for content score. Default is 1.
    - beta (float, optional): 
        Weight for novelty score. Default is 1.
    - gamma (float, optional): 
        Weight for position score. Default is 1.
    - normalize_weights_fn (function, optional): 
        Function to normalize the weights. Default is normalize_weights.

    Returns:
    - tensor: 
        Overall scores for each sentence.
    """
    # Get scores for each evalutaion
    content_scores = sentence_content_score(sents_emb)
    novelty_scores = sentence_novelty_score(sents_emb, content_scores, tau)
    position_scores = sentence_position_score(sents_emb)
    
    # Weighted sum of the scores
    alpha, beta, gamma = normalize_weights_fn(torch.tensor([alpha, beta, gamma]))
    scores = alpha * content_scores.detach().cpu() + beta * novelty_scores.detach().cpu() + gamma * position_scores.detach().cpu()
    return scores