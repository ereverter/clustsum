#!/usr/bin/python
"""
evaluate.py
"""

import argparse
import time
import os
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import evaluate
from src.clustsum import clustsum
from src.embeddings import forward_fn_pooler, forward_fn_cls
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

disable_progress_bar()

def fetch_args():
    """
    Fetches the arguments from the command line.

    Returns:
    args (argparse.Namespace):
        arguments
    """
    parser = argparse.ArgumentParser(description='Evaluate the unsupervised summaries on the chosen dataset.')
    
    # Model Details
    parser.add_argument('--method', type=str, default='transformer', help='method to use for summarization')
    parser.add_argument('--checkpoint', type=str, default='microsoft/deberta-v3-base', help='checkpoint to use')
    parser.add_argument('--embedding_from', type=str, default='cls', help='embedding to use from the transformer model')
    parser.add_argument('--max_length', type=int, default=512, help='maximum length of the input')
    parser.add_argument('--language', type=str, default='en_core_web_sm', help='language model to use')

    # Dataset Details
    parser.add_argument('--dataset', type=str, default='cnn_dailymail', choices=['cnn_dailymail', 'pubmed'], help='dataset to use')
    parser.add_argument('--subset', type=int, default=0, help='subset of the dataset to use')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size to use')

    # Processing and Hardware
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='device to use')

    # Hyperparameters
    parser.add_argument('--sum_size', type=int, default=3, help='number of sentences in the summary')
    parser.add_argument('--tau', type=float, default=0.95, help='tau for the scoring function')
    parser.add_argument('--alpha', type=float, default=0.6, help='alpha for the scoring function')
    parser.add_argument('--beta', type=float, default=0.2, help='beta for the scoring function')
    parser.add_argument('--gamma', type=float, default=0.2, help='gamma for the scoring function')

    args = parser.parse_args()
    return args

def process_batched(texts, method, args, tokenizer=None, model=None, forward_fn=None, return_scores=False):
    """
    Process the previously batched texts.
    """
    summaries = []
    for batch in tqdm(texts, desc="Processing batches"):
        sents_list = clustsum(batch, method, args, tokenizer=tokenizer, model=model, forward_fn=forward_fn, return_scores=return_scores)
        for sents in sents_list:
            summaries.append('. '.join(sents[:args.sum_size]))
    return summaries

if __name__ == '__main__':
    # Fetch the arguments
    args = fetch_args()  # Configuration

    # Load the dataset
    print(f"Loading the {args.dataset} dataset...")
    if args.dataset == 'cnn_dailymail':
        dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')
        text_column = 'article'
        summary_column = 'highlights'
    elif args.dataset == 'pubmed':
        # Adjust the path for 'pubmed' as needed.
        dataset = load_dataset('ccdv/pubmed-summarization', split='test')
        text_column = 'article'
        summary_column = 'abstract'
    else:
        raise Exception("Please provide a valid dataset.")

    # Subset the dataset
    if args.subset > 0:
        dataset = dataset.select(range(args.subset))

    # Get the summaries
    print(f"Getting the summaries with the {args.method}-based method...")
    start_time = time.time()
    if args.method == 'transformer':
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        model = AutoModel.from_pretrained(args.checkpoint)
        model.to(args.device)

        # Set the forward function
        if args.embedding_from == 'pooler':
            forward_fn = forward_fn_pooler
        elif args.embedding_from == 'cls':
            forward_fn = forward_fn_cls
        else:
            raise Exception("Please provide a valid method for the forward pass.")

        # Get the summaries in a batched manner
        batches = [dataset[i:i+args.batch_size][text_column] for i in range(0, len(dataset), args.batch_size)]
        summaries = process_batched(batches, args.method, args, tokenizer=tokenizer, model=model, forward_fn=forward_fn, return_scores=False)

    elif args.method == 'compression':
        sents = clustsum(dataset[text_column], args.method, return_scores=False)
        summaries = ['. '.join(sents[i][:args.sum_size]) for i in range(len(sents))]

    else:
        raise Exception("Please provide a valid method, either 'transformer' or 'compression'.")
    end_time = time.time()
    # Compute the ROUGE scores
    print("Computing the ROUGE scores...")
    rouge = evaluate.load('rouge')
    scores = rouge.compute(predictions=summaries, references=[item[summary_column] for item in dataset], use_aggregator=True)

    # Print the results
    print(f"ROUGE-1: {scores['rouge1']}")
    print(f"ROUGE-2: {scores['rouge2']}")
    print(f"ROUGE-L: {scores['rougeL']}")
    print(f"Time taken: {end_time - start_time}")

    # Save the results
    print("Saving the results...")
    if not os.path.exists('results'):
        os.makedirs('results')

    # Save the results differently depending on the method
    if args.method == 'transformer':
        with open(f"results/{args.dataset}_{args.method}_{args.embedding_from}_{args.checkpoint.split('/')[-1]}_{args.sum_size}_{args.tau}_{args.alpha}_{args.beta}_{args.gamma}.txt", 'w') as f:
            f.write(f"ROUGE-1: {scores['rouge1']}\n")
            f.write(f"ROUGE-2: {scores['rouge2']}\n")
            f.write(f"ROUGE-L: {scores['rougeL']}\n")
            f.write(f"Time taken: {end_time - start_time}\n")
    elif args.method == 'compression':
        with open(f"results/{args.dataset}_{args.method}_{args.sum_size}_{args.tau}_{args.alpha}_{args.beta}_{args.gamma}.txt", 'w') as f:
            f.write(f"ROUGE-1: {scores['rouge1']}\n")
            f.write(f"ROUGE-2: {scores['rouge2']}\n")
            f.write(f"ROUGE-L: {scores['rougeL']}\n")
            f.write(f"Time taken: {end_time - start_time}\n")