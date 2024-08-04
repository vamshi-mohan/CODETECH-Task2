# CODETECH-Task2
- **Name**: KAVADE VAMSHI MOHAN
- **Company**: CODETECH IT SOLUTIONS
- **ID**: CT08DS4246
- **Domain**: MACHINE LEARNING
- **Duration**: July to August 2024
- **Mentor**: Muzammil Ahmed
# IMDB Sentiment Analysis

## Overview
This project involves sentiment analysis of movie reviews from the IMDB dataset. It includes data loading, preprocessing, feature extraction, model training, and evaluation using various machine learning techniques.


## Libraries
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical operations
- `matplotlib`, `seaborn` - Data visualization
- `nltk` - Natural language processing
- `scikit-learn` - Machine learning algorithms
- `wordcloud` - Generating word clouds

## Data Loading
The project uses the IMDB movie review dataset. Basic information and sample reviews are displayed for initial analysis.

## Data Analysis
- Displayed sentiment distribution.
- Examined and visualized word count and review length.
- Processed reviews by removing HTML tags, URLs, and special characters.
- Applied stemming and removed duplicate entries.

## Text Preprocessing
- Tokenized text and removed stopwords.
- Applied stemming to the tokenized text.

## Visualization
- Created word clouds and bar charts for frequent words in positive and negative reviews.

## Feature Extraction
- Used TF-IDF Vectorizer to convert text data into numerical features.

## Model Training and Evaluation
- Split data into training and test sets.
- Trained and evaluated three models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Classifier (SVC)
- Performed hyperparameter tuning for SVC using GridSearchCV.

## Suggestions for Improvements or Extensions
- **Text Processing**: Consider lemmatization and advanced text cleaning techniques.
- **Feature Engineering**: Experiment with word embeddings (Word2Vec, GloVe) or deep learning-based embeddings (BERT).
- **Model Evaluation**: Include cross-validation and use additional metrics like ROC-AUC and Precision-Recall curves.
- **Error Analysis**: Analyze misclassified reviews to improve model performance.
- **Model Deployment**: Deploy the model as a web service or integrate it into an application for real-time analysis.
- **Documentation**: Add comments and docstrings for code clarity and create a detailed README.md.

## Installation
```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn wordcloud
