# Kindle Review Sentiment Analysis

This project performs **Binary Sentiment Classification** on a dataset of Amazon Kindle reviews. The goal is to classify a given review text as either **Positive (1)** or **Negative (0)** using a **Gaussian Naive Bayes** classifier. The project compares the performance of three popular text vectorization techniques: **Bag of Words (BoW), TF-IDF, and Word2Vec**.

---

## Project Structure

- `kindleReview.ipynb`: The main Jupyter Notebook containing data loading, preprocessing, modeling, and evaluation steps.  
- `all_kindle_review.csv` (assumed): The input dataset containing the Kindle reviews.

---

## Dataset & Target Variable

The analysis uses two key columns from the dataset:

- `reviewText`: The text content of the review.  
- `rating`: The star rating (1 to 5).

### Binary Classification Target

The original 1-5 star ratings were converted into a **binary target variable**:

- **Positive (1)**: Ratings ≥ 3  
- **Negative (0)**: Ratings < 3  

The resulting dataset is **imbalanced**, consisting of approximately 8000 positive and 4000 negative reviews.

---

## Text Preprocessing

A comprehensive text cleaning and preparation pipeline was applied to the `reviewText` column:

1. **Lowercasing**  
2. **Noise Removal**: Removing special characters, URLs, and HTML tags  
3. **Stopword Removal**: Eliminating common English stopwords (e.g., "the", "a", "is")  
4. **Text Normalization (Lemmatization)**: Reducing words to their base or root form using `WordNetLemmatizer`  
5. **Space Normalization**: Removing extra spaces  

---

## Feature Engineering (Vectorization)

The cleaned review text was converted into numerical vectors using three distinct methods:

1. **Bag of Words (BoW)**  
   - Uses `CountVectorizer` where each feature is a word count.

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**  
   - Uses `TfidfVectorizer` to weight word counts, emphasizing words that are unique or important to a specific review.

3. **Word2Vec (W2V) - Averaging**  
   - A Word2Vec model was trained on the training data with a vector size of 100.  
   - The vector for each review was computed by taking the **average of the Word2Vec embeddings** of all its words.

---

## Modeling and Results

A **Gaussian Naive Bayes (GaussianNB)** classifier was used for sentiment prediction on an 80/20 train-test split.

| Vectorization Technique | Classifier | Test Accuracy |
|-------------------------|------------|---------------|
| Bag of Words (BoW)      | GaussianNB | ≈ 0.5788      |
| TF-IDF                  | GaussianNB | ≈ 0.5750      |
| Word2Vec (Averaging)    | GaussianNB | **≈ 0.6504**  |

> The **Word2Vec (Averaging)** approach provided the highest accuracy due to its ability to capture semantic context.

---

## Requirements

Python libraries required:

- `pandas`  
- `numpy`  
- `scikit-learn`  
- `nltk`  
- `beautifulsoup4`  
- `gensim`  
- `lxml`  

---

## Installation

```bash
pip install pandas numpy scikit-learn nltk beautifulsoup4 gensim lxml
