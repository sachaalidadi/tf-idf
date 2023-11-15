# TF-IDF

This repository contains the code for the TF-IDF algorithm.

The algorithm (short for term frequency–inverse document frequency), is a measure of importance of a word to a document in a collection or corpus.

The importance of a word is measured like this:

$tfidf_{i,j} = tf_{i,j} \times idf_i$

Where $tf$ is the term frequency, and $idf$ is the inverse document frequency.

The formula of $tf$ is the following:

$$tf_{i,j} = \frac{n_{i,j}}{\sum_k n_{k,j}}$$

Where $n_{i,j}$ is the number of occurrences of the word $i$ in the document $j$, and $\sum_k n_{k,j}$ is the sum of the occurrences of all the words in the document $j$.

The formula of $idf$ is the following:

$$idf_i = \log \frac{|D|}{|\{j:t_i \in d_j\}|}$$

Where $|D|$ is the number of documents, and $|\{j:t_i \in d_j\}|$ is the number of documents where the word $i$ appears.

## Usage

The algorithm is implemented in the `tfidf.py` file. It can be used like this:

```python3 tfidf.py```

If you have a text file to test the program you can add an argument to the command:

```python3 tfidf.py [PATH]/[Name of the file]```