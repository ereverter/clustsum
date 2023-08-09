#usr/bin/bash

python eval.py \
    --subset 5000 \
    --device 'cuda' \
    --alpha 0.6 \
    --beta 0.2 \
    --gamma 0.2 \
    --tau 0.95 \
    --method 'transformer' \
    --checkpoint 'sentence-transformers/all-mpnet-base-v2' \
    --max_length 384 \
    --embedding_from 'pooler' \
    --dataset 'cnn_dailymail' \
    --sum_size 3