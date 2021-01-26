# -*- coding: utf-8 -*-
"""
modelling

"""

########################################## DEPENDENCIES ###########################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import RidgeClassifier
import nltk
import string
import re

#Data Preprocessing and Feature Engineering
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from gensim.models import word2vec
from gensim.models.fasttext import FastText
from nltk.stem.wordnet import WordNetLemmatizer
#from glove import Glove
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")
# set figure size
from pylab import rcParams
rcParams['figure.figsize'] = 13, 9

# import Counter function
from collections import Counter 
from sklearn import svm
from wordcloud import WordCloud

from gensim.models.fasttext import FastText
from gensim.models.ldamulticore import LdaMulticore
from gensim.corpora import Dictionary

from pprint import pprint

from sklearn.preprocessing import MinMaxScaler, StandardScaler

########################################## FUNCTIONS ##############################################

def load_dataset(dataset):
    df=pd.read_csv(dataset)
    return df

def tfidf_train_val(train, val, min_frequency):
    tfidf_vectorizer = TfidfVectorizer(min_df=min_frequency)

    train_tfidf = tfidf_vectorizer.fit_transform(train)
    val_tfidf  = tfidf_vectorizer.transform(val)

    return train_tfidf, val_tfidf, tfidf_vectorizer

def tfidf_test(data, min_frequency):
    tfidf_vectorizer = TfidfVectorizer(min_df=min_frequency)

    data_tfidf = tfidf_vectorizer.fit_transform(data)

    return data_tfidf, tfidf_vectorizer

# need to Generate a format of ‘ list of lists’ to use word2vec
def build_corpus(x):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for text in x.iteritems():
        sentence_list = text[1].split(".")
        words = []
        for sentence in sentence_list:
            word_list = sentence.split(" ")
            words.append(word_list)
        corpus.append(words)
    # remove empty strings from final list
    final_list = []
    for i in range(len(corpus)):
        a = corpus[i]
        for j in range(len(a)):
            corpus[i][j] = list(filter(None, corpus[i][j]))
            final_list.append(corpus[i][j])
    return final_list

# cross-validation function
# train different models and do cross validation with upsampled training set (X_res and y_res)
# HINT: Use boxplots and stripplot below to compare the models
def cross_validation_pipeline(X_train_vectorized, title):
    
    # define ML models
    models = [
        RidgeClassifier(),
        RandomForestClassifier(random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    # set number of folds (3, 5 usually)
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    # set scoring (metric performance)
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X_train_vectorized, y_train, scoring='accuracy', cv=CV)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
            cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1'])
    sns.boxplot(x='model_name', y='f1', data=cv_df)
    sns.stripplot(x='model_name', y='f1', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.savefig('figures/cross_val_'+title+'.png')
    plt.show()

# visualize embeddings
def tsne_plot(model,perplexity=10):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     fontsize=14,
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

# averaging vectors function for word2vec, GloVe and Fast text embeddings
def average_word_vectors(words, model, vocabulary, num_features):
    
    feature_vector = np.zeros((num_features,),dtype="float64")
    nwords = 0.
    
    for word in words:
        if word in vocabulary: 
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])
    
    if nwords:
        feature_vector = np.divide(feature_vector, nwords)
        
    return feature_vector  
   
def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                    for tokenized_sentence in corpus]
    return np.array(features)

def word_length_distribution(corpus):
    lengths = [len(i) for i in corpus]
    fig, ax = plt.subplots(figsize=(15,6))
    ax.set_title("Distribution of number of words", fontsize=16)
    ax.set_xlabel("Number of words", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.tick_params(labelsize=14)
    sns.distplot(lengths, bins=np.array(lengths).max(), ax=ax)

def word_frequency_barplot(corpus, nb_top_words=30):
    '''
    function to plot most frequent words' frequency from a corpus
    '''
    # flatten corpus to get a single list containing all the words
    corpus_flattened=[]
    for el in corpus:
        corpus_flattened.append(el[0])
    
    counter=Counter(corpus_flattened)
    most_occur=counter.most_common(nb_top_words)
    
    # get words and their occurences
    most_frequent_words=list(zip(*most_occur))[0]
    occurences=list(zip(*most_occur))[1]
    most_frequent_words=list(most_frequent_words)
    occurences=list(occurences)
    
    # plot 
    fig, ax = plt.subplots(1,1,figsize=(20,5))
    sns.barplot(most_frequent_words, occurences, palette='hls', ax=ax)
    ax.set_xticks(list(range(nb_top_words)))
    ax.set_xticklabels(most_frequent_words, fontsize=14, rotation=60)
    
    return ax

# wordcloud visualization function
def word_cloud(text):
    
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=50, max_words=300, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# compute average lenght of tweet
# argument needs to be list of lists
def averageLen(lst):
    lengths = [averageLen(i) for i in my_list]
    return 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths))


def build_corpus(x):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for text in x.iteritems():
        sentence_list = text[1].split(".")
        words = []
        for sentence in sentence_list:
            word_list = sentence.split(" ")
            words.append(word_list)
        corpus.append(words)
    # remove empty strings from final list
    final_list = []
    for i in range(len(corpus)):
        a = corpus[i]
        for j in range(len(a)):
            corpus[i][j] = list(filter(None, corpus[i][j]))
            final_list.append(corpus[i][j])
    return final_list

def get_topic_top_words(lda_model, topic_id, nr_top_words=5):
    """ Returns the top words for topic_id from lda_model.
    """
    id_tuples = lda_model.get_topic_terms(topic_id, topn=nr_top_words)
    word_ids = np.array(id_tuples)[:,0]
    words = map(lambda id_: lda_model.id2word[id_], word_ids)
    return words

def document_to_lda_features(lda_model, document):
    """ Transforms a bag of words document to features.
    It returns the proportion of how much each topic was
    present in the document.
    """
    topic_importances = LDAmodel.get_document_topics(document, minimum_probability=0)
    topic_importances = np.array(topic_importances)
    return topic_importances[:,1]

# Showing Confusion Matrix
def plot_cm(y_true, y_pred, title, figsize=(5,4)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.savefig('figures/confusion_matrix.png')
    plt.show()


def visualize_word_similarities(relevant_words):
    
    similar_words = {search_term: [item[0] for item in model_fast_text.wv.most_similar([search_term], topn=7)]
                  for search_term in relevant_words}
    words = sum([[k] + v for k, v in similar_words.items()], [])
    wvs = model_fast_text.wv[words]
    pca = PCA(n_components=2)
    np.set_printoptions(suppress=True)
    P = pca.fit_transform(wvs)
    labels = words
    plt.figure(figsize=(18, 10))
    plt.scatter(P[:, 0], P[:, 1], c='lightgreen', edgecolors='g')
    for label, x, y in zip(labels, P[:, 0], P[:, 1]):
        plt.annotate(label, xy=(x+0.06, y+0.03), xytext=(10, 10), fontsize=14, textcoords='offset points')
    plt.show()

########################################## EXECUTION ###########################################

# load cleaned data
df_train_cleaned=load_dataset('data/train_cleaned.csv')
df_test_cleaned=load_dataset('data/test_cleaned.csv')
df_train_cleaned.dropna(inplace=True)
df_test_cleaned.fillna('-', inplace=True)

# split into train/val sets
X=df_train_cleaned['cleaned_text']
y=df_train_cleaned['target']
X_test=df_test_cleaned['cleaned_text']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=0)
print('shape training set: %d' % X_train.shape[0])
print('shape validation set: %d' % X_val.shape[0])

### Vectorizing tweets ###

# Bag of words method
min_word_frequency=10
count_vectorizer = CountVectorizer(min_df=min_word_frequency)
X_train_bow=count_vectorizer.fit_transform(X_train)
print('There are %d unique words in all the tweets' % X_train_bow[0].todense().shape[1])
X_test_bow=count_vectorizer.transform(X_test)

# TF-IDF method
min_word_frequency=10
X_train_tfidf, X_val_tfidf, tfidf_vectorizer = tfidf_train_val(X_train, X_val, min_word_frequency)
X_test_tfidf=tfidf_vectorizer.transform(X_test)
print('There are %d unique words in all the tweets' % X_train_tfidf.shape[1])

# Word2vec method
X_train_corpus = build_corpus(X_train) 
len(X_train_corpus)
feature_size_w2vec=100
window_w2vec=15
min_count_w2vec=40
workers_w2vec=4
model_w2vec = word2vec.Word2Vec(X_train_corpus, size=feature_size_w2vec, window=window_w2vec, min_count=min_count_w2vec, workers=workers_w2vec)
words_w2vec=list(model_w2vec.wv.vocab)
print('vocabulary size: %d' % len(words_w2vec))

# testing model: find the most similar words of a given word or a combination of words
print('words similar to disaster: ', model_w2vec.most_similar('disaster'))
print('words similar to disaster, accident, emergency: ', model_w2vec.most_similar_cosmul(positive=['disaster', 'accident', 'emergency']))

# visualize word similarities
tsne_plot(model_w2vec,15)

# save word embeddings
filename='models/twitter_disaster_word2vec.txt'
model_w2vec.wv.save_word2vec_format(filename, binary=False)

# Fast Text method 
feature_size_ftext = 100    # Word vector dimensionality  
window_ftext = 10          # Context window size                                                                                    
min_count_ftext = 20   # Minimum word count
sample = 1e-3   # Downsample setting for frequent words
model_fast_text = FastText(X_train_corpus, size=feature_size_ftext, window=window_ftext, 
                    min_count=min_count_ftext,sample=sample, sg=1, iter=50)
visualize_word_similarities(['disaster', 'accident', 'emergency', 'bomb', 'fire'])



# Testing Machine Learning models
cross_validation_pipeline(X_train_bow, 'BOW')
cross_validation_pipeline(X_train_tfidf, 'TFIDF')

# Model performance
# fit model
NB=MultinomialNB()
NB.fit(X_train_tfidf, y_train)
y_pred=NB.predict(X_val_tfidf)
plot_cm(y_pred, y_val, 'Confusion matrix', figsize=(7,7))

# choose vectorization method
min_word_frequency=10
X_train_tfidf, tfidf_vectorizer = tfidf_test(X_train, min_word_frequency)
X_val_tfidf=tfidf_vectorizer.transform(X_val)
# fit model
NB=MultinomialNB()
NB.fit(X_train_tfidf, y_train)
# predictions
y_pred=NB.predict(X_val_tfidf)

# get word2vec embeddings as features
X_train_w2vec = averaged_word_vectorizer(corpus=X_train_corpus, model=model_w2vec,
                                             num_features=feature_size_w2vec)

# get FastText embeddings as features
X_train_ft = averaged_word_vectorizer(corpus=X_train_corpus, model=model_fast_text,
                                             num_features=feature_size_ftext)

"""# Testing pipeline"""

# choose vectorization method
min_word_frequency=10
X_tfidf, tfidf_vectorizer = tfidf_test(X, min_word_frequency)
X_test_tfidf=tfidf_vectorizer.transform(X_test)
# fit model
NB=MultinomialNB()
NB.fit(X_tfidf, y)
# predictions
y_pred=NB.predict(X_test_tfidf)

# add predictions to dataframe
results=pd.DataFrame(df_test_cleaned['id'])
results['predictions']=y_pred

min_word_frequency=10
count_vectorizer = CountVectorizer(min_df=min_word_frequency)
X_bow=count_vectorizer.fit_transform(X)
X_test_bow=count_vectorizer.transform(X_test)
# fit model
log=LogisticRegression(random_state=0)
log.fit(X_bow, y)
# predictions
y_pred=NB.predict(X_test_bow)

# save results
results.to_csv('models/results_tfidf.csv',index=False)

"""## using word2vec"""

X_corpus = build_corpus(X) 
X_test_corpus = build_corpus(X_test) 
feature_size_w2vec=100
window_w2vec=10
min_count_w2vec=30
workers_w2vec=4
model_w2vec = word2vec.Word2Vec(X_corpus, size=feature_size_w2vec, window=window_w2vec, min_count=min_count_w2vec, workers=workers_w2vec)

# get word2vec embeddings as features
X_w2vec = averaged_word_vectorizer(corpus=X_corpus, model=model_w2vec,
                                             num_features=feature_size_w2vec)
X_test_w2vec=averaged_word_vectorizer(corpus=X_test_corpus, model=model_w2vec,
                                             num_features=feature_size_w2vec)

scaler=MinMaxScaler()
X_w2vec = scaler.fit_transform(X_w2vec)
X_test_w2vec = scaler.transform(X_test_w2vec)

# fit model
NB=MultinomialNB()
NB.fit(X_w2vec, y)
# predictions
y_pred=NB.predict(X_test_w2vec)

# add predictions to dataframe
results_w2vec=pd.DataFrame(df_test_cleaned['id'])
results_w2vec['target']=y_pred
results_w2vec.head(50)

results_w2vec['target'].value_counts()

# save results
results_w2vec.to_csv('models/w2vec_results.csv',index=False)