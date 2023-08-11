#usr/bin/python
"""
clustsum.py
"""

import spacy
from transformers import AutoTokenizer, AutoModel
from .embeddings import get_embeddings, get_compression_distance, forward_fn_cls
from .utils import text_to_sentences
from .scoring import sentence_scores
from .config import Configuration

def clustsum(texts, method='transformer', config=None, nlp=None, tokenizer=None, model=None, forward_fn=forward_fn_cls, return_scores=False):
    """
    Summarize input texts using an unsupervised extractive approach.

    Args:
    - texts (List[str]): 
        Input texts for summarization.
    - method (str, optional): 
        Embedding technique. Options are 'transformer' or 'compression'. Default is 'transformer'.
    - config (Configuration, optional): 
        Configuration object containing necessary parameters.
    - nlp (spacy Language, optional): 
        Language model for sentence tokenization.
    - tokenizer (transformer tokenizer, optional): 
        Tokenizer for the transformer model.
    - model (transformer model, optional): 
        Transformer model used for sentence embeddings.
    - forward_fn (function, optional): 
        Function to obtain embeddings from the transformer model.
    - return_scores (bool, optional): 
        Whether to return sentence scores. Default is False.

    Returns:
    - List[str]: If return_scores is False, returns summarized sentences.
    - Tuple[List[str], tensor]: If return_scores is True, returns tokenized sentences and their scores.
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

    print('o')
    # Preprocess each text into sentences
    all_sents = [text_to_sentences(text, nlp) for text in texts]
    print(all_sents)

    # Get the embeddings for all samples
    if method == 'transformer':
        all_sents_emb = get_embeddings(all_sents, tokenizer, model, forward_fn, config)
    elif method == 'compression':
        all_sents_emb = get_compression_distance(all_sents)
    
    # Compute the scores
    sents_scores = [sentence_scores(sents_emb, config.tau, config.alpha, config.beta, config.gamma) for sents_emb in all_sents_emb]

    # Output (sentences and scores) or sorted sentences 
    if return_scores:
        return all_sents, sents_scores
    return [[sents[idx] for idx in sents_scores[i].argsort(descending=True)] for i, sents in enumerate(all_sents)]
