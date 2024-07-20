# Hybrid Text Summarization

## Text Preprocessing and Summarization Using TextRank and K-Means
Libraries and Tools
NLP Libraries: nltk, re for text preprocessing, networkx for graph-based ranking.
Machine Learning: sklearn for clustering and vector operations.
Metrics and Evaluation: evaluate and load for ROUGE metric evaluation.
Visualization: matplotlib, seaborn for plotting.
Text Summarization: transformers library with BART model for final summarization.
Key Functions and Methods
Word Embeddings Extraction:

Extracts word embeddings using GloVe vectors to convert words into numerical vectors.
Text Preprocessing:

Cleans text by removing non-alphabetic characters and stopwords.
Tokenizes sentences and words, and lemmatizes them for normalization.
Sentence Vector Representation:

Creates sentence vectors by averaging the word vectors in each sentence.
Similarity Matrix Creation:

Generates a similarity matrix using cosine similarity to measure the similarity between sentences.
Sentence Ranking:

Uses the PageRank algorithm on the similarity matrix to rank sentences based on their importance.
Summary Generation Using TextRank:

Generates a summary by selecting the top-ranked sentences based on the TextRank algorithm.
Summary Refinement Using K-Means Clustering:

Clusters sentence vectors and selects representative sentences from each cluster to refine the summary.
Utilizes Euclidean distance to find the nearest sentences to cluster centroids.
Final Summary Generation with BART Model:

Uses the BART model from transformers library to generate a final summary by handling longer texts in chunks.

## Detailed Workflow
Extract Word Vectors:

Load pre-trained GloVe embeddings and create a dictionary of word vectors.
Text Preprocessing:

Clean and tokenize input text, remove stopwords, and lemmatize words.
Create Sentence Vectors:

For each sentence, compute its vector representation by averaging the vectors of words in the sentence.
Create Similarity Matrix:

Compute a similarity matrix using cosine similarity to measure the pairwise similarity between sentences.
Rank Sentences with TextRank:

Construct a graph from the similarity matrix and apply the PageRank algorithm to rank sentences.
Generate Initial Summary with TextRank:

Select the top-ranked sentences to form an initial summary.
Refine Summary with K-Means Clustering:

Perform K-means clustering on sentence vectors, then select the closest sentences to cluster centroids to include in the summary.
Generate Final Summary with BART:

Tokenize the cleaned and summarized text, process it in chunks, and use the BART model to generate a coherent final summary.
