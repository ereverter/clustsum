#usr/bin/python
"""
scoring.py
"""

import torch
from torch.nn.functional import cosine_similarity as cos
from .utils import normalize_weights

def sentence_content_score(sents_emb):
    """
    Computes content scores for sentences by computing cosine similarity with the centroid.

    Args:
    sents_emb: tensor, sentence embeddings

    Returns:
    tensor, content scores
    """
    centroid = sents_emb.mean(dim=0)
    return cos(sents_emb, centroid, dim=1)

def sentence_novelty_score(sents_emb, content_scores, tau):
    """
    Computes novelty scores for sentences based on their similarity with other sentences.

    Args:
    sents_emb: tensor, sentence embeddings
    content_scores: tensor, content scores
    tau: float, threshold for novelty score computation

    Returns:
    novelty_scores: np.ndarray, novelty scores
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
    Computes position scores for sentences based on their position in the text.

    Args:
    sents_emb: tensor, sentence embeddings

    Returns:
    scores: tensor, position scores
    """
    M = sents_emb.shape[0]
    sentence_positions = torch.tensor([i for i in range(M)])
    scores = torch.max(torch.tensor(0.5), torch.exp(-sentence_positions / (3 * torch.sqrt(torch.tensor(M, dtype=torch.float)))))
    return scores

def sentence_scores(sents_emb, tau=0.5, alpha=1, beta=1, gamma=1, normalize_weights_fn=normalize_weights):
    """
    Computes overall scores for sentences based on content, novelty, and position scores.

    Args:
    sents_emb: tensor, sentence embeddings
    tau: float, threshold for novelty score computation
    alpha: float, weight for content score
    beta: float, weight for novelty score
    gamma: float, weight for position score
    normalize_weights_fn: function, normalization function

    Returns:
    scores: tensor, overall scores
    """
    # Get scores for each evalutaion
    content_scores = sentence_content_score(sents_emb)
    novelty_scores = sentence_novelty_score(sents_emb, content_scores, tau)
    position_scores = sentence_position_score(sents_emb)
    
    # Weighted sum of the scores
    alpha, beta, gamma = normalize_weights_fn(torch.tensor([alpha, beta, gamma]))
    scores = alpha * content_scores.detach().cpu() + beta * novelty_scores.detach().cpu() + gamma * position_scores.detach().cpu()
    return scores