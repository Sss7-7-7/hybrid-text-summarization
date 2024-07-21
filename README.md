# Hybrid Text Summarization

## Text Preprocessing and Summarization Using TextRank, K-Means and BART model
Libraries and Tools

• NLP Libraries: 'nltk', 're' for text preprocessing, networkx for graph-based ranking.

• Machine Learning: 'sklearn' for clustering and vector operations.

• Metrics and Evaluation: 'evaluate' and 'load' for ROUGE metric evaluation.

• Visualization: 'matplotlib', 'seaborn' for plotting.

• Text Summarization: 'transformers' library with BART model for final summarization.

## Summary

• Summarization involves extracting key information while maintaining coherence. Pre-processing is crucial for success.

• Pre-processing Stage: Text cleaning removes irrelevant characters and formatting artifacts. Sentence tokenization breaks text into sentences. Word tokenization dissects sentences into words for analysis. Stopword removal eliminates common words. Lemmatization or stemming standardizes word forms. Part-of-speech tagging categorizes words grammatically. Named Entity Recognition identifies entities for coherence.

• Additional Pre-processing Steps: Handling linguistic nuances, contractions, and punctuation marks. Feature engineering includes sentence length and word frequency.

• Extractive Text Summarization: Reads pre-processed text from top to bottom. Selects important sentences for an extractive summary.

• Abstractive Text Summarization: Takes extractive summary as input. Identifies weightage words using predefined datasets. Generates a new summary based on word meanings.

• Summary Evaluation: Evaluates the generated summary for accuracy. If accuracy is high, the summary is considered good. Adjustments are made until the accuracy is satisfactory.

## Detailed Workflow

1. Extract Word Vectors:

• Load pre-trained GloVe embeddings and create a dictionary of word vectors.

2. Text Preprocessing:

• Clean and tokenize input text, remove stopwords, and lemmatize words.

3. Create Sentence Vectors:

• For each sentence, compute its vector representation by averaging the vectors of words in the sentence.

4. Create Similarity Matrix:

• Compute a similarity matrix using cosine similarity to measure the pairwise similarity between sentences.

5. Rank Sentences with TextRank:

• Construct a graph from the similarity matrix and apply the PageRank algorithm to rank sentences.

6. Generate Initial Summary with TextRank:

• Select the top-ranked sentences to form an initial summary.

7. Refine Summary with K-Means Clustering:

• Perform K-means clustering on sentence vectors, then select the closest sentences to cluster centroids to include in the summary.

8. Generate Final Summary with BART:

• Tokenize the cleaned and summarized text, process it in chunks, and use the BART model to generate a coherent final summary.

## Necessary Libraries:
nltk: Needed for tokenizing.
```bash
import nltk  # import nlp toolkit library
nltk.download('punkt')  # one-time download
nltk.download('stopwords')  # one-time download
```

## Credits

This project was developed with the assistance of various resources and tools, including:

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) - Used for word embeddings.
- [NetworkX](https://networkx.github.io/) - Used for graph-based ranking algorithms.
- [NLTK](https://www.nltk.org/) - Used for text preprocessing and tokenization.
- [scikit-learn](https://scikit-learn.org/) - Used for cosine similarity and K-Means clustering.
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Used for BART model implementation.
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) - Used for plotting and visualization.

Special thanks to [ChatGPT by OpenAI](https://www.openai.com/research/chatgpt) for providing assistance with generating and refining the content of this project.

## Acknowledgments

- Special thanks to [[Sathyanarayana-NITK](https://github.com/Sathyanarayana-NITK)] for the guidance.
