#usr/bin/python
"""
config.py
"""

class Configuration:
    language = 'en_core_web_sm'
    tau = 0.75
    alpha = 2
    beta = 1
    gamma = 1
    max_length = 512
    checkpoint = 'microsoft/deberta-v3-base'
    embedding_from = 'cls' # 'pooler' or 'cls'
    method = 'transformer' # 'transformer' or 'compression'
    device = 'cpu'
    subset = 10