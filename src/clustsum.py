#usr/bin/python
"""
clustsum.py
"""

import spacy
from transformers import AutoTokenizer, AutoModel
from .embeddings import get_embeddings, get_compression_distance, get_batched_embeddings
from .utils import text_to_sentences
from .scoring import sentence_scores
from .config import Configuration

def clustsum(texts, method='transformer', config=None, nlp=None, tokenizer=None, model=None, forward_fn=None, return_scores=False):
    """
    Main function to apply unsupervised extractive summarization.

    Args:
    text: str, input text
    method: str, method to use for summarization, either 'transformer' or 'compression'
    nlp: spacy Language, language model for tokenization
    tokenizer: transformer tokenizer, tokenizer for the transformer model
    model: transformer model, transformer model for sentence embedding
    forward_fn: function, function to get embeddings from the transformer model

    Returns:
    sents: list of str, tokenized sentences
    sents_scores: tensor, scores for sentences
    """
    # Load everything that is needed
    if config is None:
        config = Configuration()        
    if nlp is None:
        nlp = spacy.load(config.language)
    if tokenizer is None and method == 'transformer':
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
    if model is None and method == 'transformer':
        model = AutoModel.from_pretrained(config.checkpoint)
    if forward_fn is None and method == 'transformer':
        raise Exception("Please provide the forward function to obtain the embedding.")

    # Preprocess each text into sentences
    all_sents = [text_to_sentences(text, nlp) for text in texts]

    # Get the embeddings for all samples
    if method == 'transformer':
        all_sents_emb = get_embeddings(all_sents, tokenizer, model, forward_fn, config)
    elif method == 'compression':
        all_sents_emb = get_compression_distance(all_sents)
        # raise Exception("Compression-based embeddings are not supported for batched processing.")
    
    # Compute the scores
    sents_scores = [sentence_scores(sents_emb, config.tau, config.alpha, config.beta, config.gamma) for sents_emb in all_sents_emb]

    # Output (sentences and scores) or sorted sentences 
    if return_scores:
        return all_sents, sents_scores
    return [[sents[idx] for idx in sents_scores[i].argsort(descending=True)] for i, sents in enumerate(all_sents)]
