# Unsupervised Extractive Summarization Based on a Centroid Approach and Sentence Embeddings

### Overview

This repository provides an implementation of an unsupervised method for extractive multi-document summarization. The approach is based on the one proposed in [Lamsiyah et al. (2021)](https://doi.org/10.1016/j.eswa.2020.114152). It utilizes two distinct techniques for sentence embeddings: transformer-based and compression-based. Additionally, a scoring system ranks sentences according to their significance. The results are presented in the tables below.

### Embeddings

#### 1. Transformer-based Embeddings

Embeddings from pre-trained transformer models are used in this method. The following embeddings are considered:

- **Pooler Output**: If available, the pooler output embeddings from the transformer model are used.
- **[CLS] Token**: If the pooler output is not available, the [CLS] (or its equivalent) token embeddings from the transformer model are used.

Sentences are tokenized and then passed through the transformer model. These tokenized sentences are padded and truncated to a specified maximum length.

#### 2. Compression-based Embeddings

Compression-based embeddings are derived using the `gzip` compression algorithm. The distance between two sentences is ascertained by compressing the combined sentence and contrasting it with individual compressions. This method offers a similarity measure between sentences. It's an adaptation of the approach presented in [Jiang et al. (2023)](https://aclanthology.org/2023.findings-acl.426/), modified for extractive summarization.

## Scoring System

### 1. Sentence Content Relevance Score

The content relevance score for a sentence \( S_i \) in cluster \( D \) is determined using the cosine similarity between the sentence embedding vector \( \vec{S_{D_i}} \) and the centroid embedding vector \( \vec{C_D} \):

```
$$ \text{score}_{\text{contentRelevance}}(S_i, D) = \frac{\vec{S_{D_i}} \cdot \vec{C_D}}{||\vec{S_{D_i}}|| \cdot ||\vec{C_D}||} $$
```
Where:
- \( \vec{S_{D_i}} \) represents the embedding vector of sentence \( S_i \).
- \( \vec{C_D} \) denotes the centroid embedding vector of cluster \( D \).

### 2. Sentence Novelty Score

The novelty score for a sentence \( S_i \) in cluster \( D \) is computed as:

```
$$ 
\text{score}_{\text{novelty}}(S_i, D) = 
\begin{cases} 
1 & \text{if } \max(\text{sim}(S_i, S_k)) < \tau \\
1 & \text{if } \max(\text{sim}(S_i, S_k)) > \tau \text{ and } \text{score}_{\text{contentRelevance}}(S_i, D) > \text{score}_{\text{contentRelevance}}(S_l, D) \\
1 - \max(\text{sim}(S_i, S_k)) & \text{otherwise}
\end{cases} 
$$
```

Where:
- \( \text{sim}(S_i, S_k) \) indicates the similarity between sentence \( S_i \) and other sentences in cluster \( D \).
- \( l \) is the index of the sentence most similar to \( S_i \) in cluster \( D \).

### 3. Sentence Position Score

The position score for a sentence \( S_{d_i} \) in a document \( d \) is:

```
$$ \text{score}_{\text{position}}(S_{d_i}) = \max\left(0.5, \exp\left(\frac{-p(S_{d_i})}{3\sqrt{M_d}}\right)\right) $$
```

Where:
- \( p(S_{d_i}) \) is the position of sentence \( S \) in document \( d \), starting from 1.
- \( M_d \) is the total number of sentences in document \( d \).

The final score for a sentence \( S_i \) is the sum of the three scores:

```
$$ \text{score}(S_i, D) = \alpha \cdot \text{score}_{\text{contentRelevance}}(S_i, D) + \beta \cdot \text{score}_{\text{novelty}}(S_i, D) + \gamma \cdot \text{score}_{\text{position}}(S_i) $$
```

Subject to:

```
$$ \alpha + \beta + \gamma = 1 $$
```

## Results

The results of both embeddings, when compared to the lead baseline (i.e., selecting the first $ m $ sentences as the summary), are displayed in the tables below. These results are derived from subsets of the [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail) and [PubMed](https://huggingface.co/datasets/pubmed) datasets. The [rouge](https://huggingface.co/spaces/evaluate-metric/rouge) metric from `evaluate` was used, which tends to yield slightly lower scores than `pyrouge` -- in my experience. After tokenizing the texts into sentences using `spacy`, the average summary length, in terms of the number of sentences, is $ m=3 $ for the former dataset and $ m=8 $ for the latter.

The transformer-based embeddings generally outperform the compression-based embeddings across all metrics. However, the latter are faster to compute, making them a suitable choice for larger datasets. It's noteworthy that the lead baseline performs better on the news dataset, where initial sentences are typically more informative than those in the article's middle. Adjusting hyperparameters for scoring and using larger dataset subsets could produce more robust results.

### cnn_dailymail (n=2500)

| Method               | Rouge-1 ↑ | Rouge-2 ↑ | Rouge-L ↑ | Time (s) ↓ |
|----------------------|-----------|-----------|-----------|------------|
| **lead**             | 34.57     | 14.54     | 22.42     | -          |
| **all-mpnet-base-v2**| 33.37     | 13.44     | 21.35     | 643        |
| **gzip**             | 32.03     | 12.49     | 21.06     | 399        |

### pubmed (n=1000)

| Method               | Rouge-1 ↑ | Rouge-2 ↑ | Rouge-L ↑ | Time (s) ↓ |
|----------------------|-----------|-----------|-----------|------------|
| **lead**             | 21.52     | 9.41      | 14.94     | -          |
| **all-mpnet-base-v2**| 41.33     | 16.91     | 21.31     | 1536       |
| **gzip**             | 36.67     | 12.24     | 19.35     | 877        |


## Usage

To evaluate the models and replicate the results, execute the following:

```bash
python clustsum/eval.py \
    --method 'transformer' \ # 'transformer' or 'compression'
    --checkpoint 'sentence-transformers/all-mpnet-base-v2' \
    --embedding_from 'pooler' \ # 'pooler' or 'cls' or define it yourself
    --max_length 384 \
    --dataset 'pubmed' \ # 'pubmed' or 'cnn_dailymail' or define it yourself
    --subset 1000 \
    --batch_size 2 \
    --device 'cuda' \ # 'cuda' or 'cpu'
    --sum_size 8 \
    --tau 0.95 \
    --alpha 0.6 \
    --beta 0.2 \
    --gamma 0.2
```

To play around with single samples, take a look at the [`playground.ipynb`](https://github.com/eReverter/clustsum/blob/main/playground.ipynb) notebook.
