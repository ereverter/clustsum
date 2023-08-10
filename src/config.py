#usr/bin/python
"""
config.py
"""

class Configuration:
    language = 'en_core_web_sm'
    tau = 0.95
    alpha = 0.6
    beta = 0.2
    gamma = 0.2
    max_length = 512
    checkpoint = 'bert-base-uncased'
    embedding_from = 'cls' # 'pooler' or 'cls'
    method = 'transformer' # 'transformer' or 'compression'
    device = 'cpu'
    subset = 10