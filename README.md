# Disaster Tweet Detector

## Project Overview

This project is a participation the Kaggle challenge [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/overview).

Twitter has become an important communication channel in times of emergency. First responders are increasingly monitoring Twitter as it gives near real-time information. But they might face difficulties to clearly identify if a person is anouncing a disaster.
With many tweets containing metaphores, the task can be tricky. 

I built a solution based on Supervised Learning that can identify if a tweet is related to a real disaster or not. This could help emergency services to automatically monitor Twitter to detect disasters with a better accuracy.

## Github Repository

This repository contains 3 scripts:
- *eda.y*: exploratory analysis of the features "keywords" and "location" to analyse possible correlations with disaster occurrence. 
- *preprocessing.py*: succession of tweet cleaning and preprocessing
- *modelling.py*: tweets vectorization (TF-IDF) and binary classification model (multinomial Naive Bayes)


## Exploratory Data Analysis

I wanted to figure out if we can leverage the columns "location" and "keyword" in the models.

Keywords analysis           |  Locations analysis   
:-------------------------:|:-------------------------:
![](/figures/keywords_analysis.png)  |  ![](/figures/locations_analysis.png)


## Tweet cleaning

This step consists in cleaning and removing the noise from the tweets, especially:
- removing URLs, digits and stop words.
- performing Part-of-speech tagging (POS) to keep only nouns, verbs and adjectives.

This allows us the have better look at the tweets' content.

Word Cloud - "Disaster" tweets          |  Word Cloud - "No disaster" tweets  
:-------------------------:|:-------------------------:
![](/figures/word_cloud_disaster_tweets.png)  |  ![](/figures/word_cloud_NO_disaster_tweets.png)

It is interesting to notice common frequent words such as "fire". Indeed, the word "fire" is greatly used in metaphores (example: "This man is on fire").  
Among the disaster tweets, we can quote these differenciators: "police", "terrorist" as well as words referring to natural disasters ("storm", "flood").


## Model building

First, I transformed the cleaned tweets into vectors.
I tried different methods (vectorizers) including:
- Count vectorizer (Bag of words: count the words occurences)
- TF-IDF (term frequency–inverse document frequency)
- Word2vec word vectors
- FastText word vectors

Then, I built the classification models with 2 classes.
I tried different models that could be adapted for our binary classification problem:
- Ridge Classifier
- Random Forest Classifier 
- LinearSVC
- Multinomial Naives-Bayes
- Logistic Regression

![](/figures/cross_val_TFIDF.png)

To improve the F1 score, so I tried with Neural Networks (see in [Notebooks](https://www.kaggle.com/c/nlp-getting-started/overview)): 
- a simple sequential model with an embedding layer (using Keras)
- ULMFit (Universal Model Fine-tuning for Text Classification) (using PyTorch/Fast.ai)
- BERT fine-tuned for text classification

## Model Performance

TF-IDF vectors + Multinomial Naive Bayes        |  BERT model
:-------------------------:|:-------------------------:
![](/figures/performance_NB.png)  |  ![](/figures/performance_BERT.png)



