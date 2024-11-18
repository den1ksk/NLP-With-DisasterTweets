#### English Version | [Kaggle competition](https://www.kaggle.com/competitions/nlp-getting-started/overview)

# NLP Text Classification for Disaster Tweets

### Competition Description

This competition focuses on classifying tweets that are related to real-world disasters. The challenge is to predict whether a tweet is about a real disaster or not based on its content. The dataset consists of 10,000 tweets that were hand-labeled as disaster-related or not, and the goal is to build a machine learning model capable of distinguishing between the two categories.

### Goal

The goal of this project is to predict whether a tweet is related to a real disaster or not. The target variable in the dataset is `target`, which takes values `1` for disaster-related tweets and `0` for non-disaster-related tweets.

### Approach

This project applies three different Natural Language Processing (NLP) techniques to classify disaster tweets. The three models explored are:

1. **NLTK-based model**
2. **spaCy-based model**
3. **BERT-based model**

#### 1. Data Exploration:
   - The dataset was explored to understand the distribution of disaster and non-disaster tweets.
   - Key features, such as tweet content, were analyzed to uncover patterns related to disaster identification.

#### 2. Data Preprocessing:
   - Text cleaning and preprocessing were done, including tokenization, lowercasing, and removal of stopwords.
   - For the NLTK and spaCy models, text vectorization techniques like TF-IDF and word embeddings were applied.
   - For the BERT model, the text was tokenized using the BERT tokenizer.

#### 3. Model Development:
   - **NLTK Model:** Classic machine learning approach using Naive Bayes and TF-IDF features.
   - **spaCy Model:** A pipeline utilizing spaCy's pre-trained embeddings and a logistic regression classifier.
   - **BERT Model:** Fine-tuned a pre-trained BERT model on the dataset for better context understanding and classification.

#### 4. Model Evaluation:
   - The models were evaluated using accuracy, precision, recall, and F1-score metrics.
   - The BERT model showed the highest performance, achieving the best classification results.

#### 5. Prediction and Submission:
   - Predictions for the test dataset were generated using each model and saved in separate CSV files for Kaggle submission.

### Results

The results showed varying performance across the models. The BERT-based model achieved the highest accuracy and F1-score, followed by the spaCy and NLTK models. Below are the performance results for each model:

- **BERT Model:** Best performance with the highest F1-score.
- **spaCy Model:** Good performance, but slightly lower than BERT.
- **NLTK Model:** Decent performance, though not as strong as spaCy and BERT.

### Files

- `bert.ipynb`: Jupyter Notebook with the BERT-based model implementation.
- `nltk.ipynb`: Jupyter Notebook with the NLTK-based model implementation.
- `spaCy.ipynb`: Jupyter Notebook with the spaCy-based model implementation.
- `submission_bert.csv`: Predictions for the test dataset using the BERT model.
- `submission_nltk.csv`: Predictions for the test dataset using the NLTK model.
- `submission_spacy.csv`: Predictions for the test dataset using the spaCy model.
- `train.csv`: The training dataset used for training the models.
- `test.csv`: The test dataset used for making predictions.

### Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- spacy
- transformers (for BERT)
- torch

### Acknowledgements

This project was completed as part of the Kaggle competition "NLP Getting Started". Special thanks to Kaggle for providing the dataset and platform for this competition!

