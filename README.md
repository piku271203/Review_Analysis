Kindle Review Sentiment Analysis
This project performs a Binary Sentiment Classification on a dataset of Amazon Kindle reviews. The goal is to classify a given review text as either Positive (1) or Negative (0) using a Gaussian Naive Bayes classifier, comparing the performance of three popular text vectorization techniques: Bag of Words (BoW), TF-IDF, and Word2Vec.

Project Structure
kindleReview.ipynb: The main Jupyter Notebook containing all the data loading, preprocessing, modeling, and evaluation steps.

all_kindle_review.csv (Assumed): The input dataset containing the Kindle reviews.

Dataset
The dataset used is sourced from Amazon Kindle reviews. The analysis focuses on two key columns:

reviewText: The actual text content of the review.

rating: The star rating (1 to 5).

Binary Classification Target
The original 1-5 star ratings were converted into a binary target variable:

Positive (1): Ratings ≥3

Negative (0): Ratings <3

The resulting dataset is imbalanced, with 8000 positive and 4000 negative reviews.

Text Preprocessing
A comprehensive text cleaning and preparation pipeline was applied to the reviewText column to ensure quality feature extraction:

Lowercasing: Converting all text to lowercase.

Noise Removal: Removing special characters, URLs, and HTML tags.

Stopword Removal: Eliminating common English stopwords (e.g., "the", "a", "is").

Text Normalization: Applying Lemmatization (using WordNetLemmatizer) to reduce words to their base or root form.

Space Normalization: Removing extra spaces.

Feature Engineering (Vectorization)
The cleaned review text was converted into numerical vectors using three different methods for comparison:

1. Bag of Words (BoW)
Uses CountVectorizer to create a feature vector where each dimension corresponds to a word in the vocabulary, and the value is the count of that word in the review.

2. TF-IDF (Term Frequency-Inverse Document Frequency)
Uses TfidfVectorizer to weight the word counts. It emphasizes words that are important to a specific review but less common across all reviews.

3. Word2Vec (W2V) - Averaging
A Word2Vec model was trained on the training data with a vector size of 100.

Each review's vector was calculated by taking the average of the Word2Vec embeddings of all the words it contains.

Modeling and Results
A Gaussian Naive Bayes (GaussianNB) classifier was used for the sentiment prediction task, trained and tested on an 80/20 train-test split.

Vectorization Technique	Classifier	Test Accuracy
Bag of Words (BoW)	GaussianNB	≈0.5788
TF-IDF	GaussianNB	≈0.5750
Word2Vec (Averaging)	GaussianNB	≈0.6504


Key Finding
The Word2Vec (Averaging) approach yielded the best performance with an accuracy of approximately 65.04%. This result is expected, as Word2Vec captures semantic relationships between words, which is often crucial for effective sentiment analysis, unlike BoW and TF-IDF which treat words in isolation.

Requirements
To run this notebook, you will need the following Python libraries:

Bash

pip install pandas numpy scikit-learn nltk beautifulsoup4 gensim lxml
You also need to download the necessary NLTK data:


import nltk
nltk.download('stopwords')
nltk.download('wordnet')
