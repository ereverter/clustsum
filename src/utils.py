#usr/bin/python
"""
utils.py
"""

import re

def clean_text(text):
    """
    Cleans the input text by removing special characters and extra spaces.
    
    Args:
    text: str, input text

    Returns:
    text: str, cleaned text
    """
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading and trailing spaces
    text = text.strip()
    return text

def text_to_sentences(text, nlp):
    """
    Tokenizes the input text into sentences.

    Args:
    text: str, input text
    nlp: spacy Language, language model for tokenization

    Returns:
    list of str, sentences
    """
    doc = nlp(text)
    return [clean_text(sent.text) for sent in doc.sents]

def normalize_weights(weights):
    """
    Normalizes the input weights to add up to 1.

    Args:
    weights: list of float, input weights

    Returns:
    tuple of float, normalized weights
    """
    return weights / weights.sum()