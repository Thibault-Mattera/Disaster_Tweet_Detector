# -*- coding: utf-8 -*-
"""
preprocessing

"""

########################################## DEPENDENCIES ###########################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import string
import re
from textblob import TextBlob
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from wordcloud import WordCloud
from tqdm import tqdm
tqdm.pandas()
from pylab import rcParams
rcParams['figure.figsize'] = 13, 9

########################################### FUNCTIONS ##############################################


# load dataset
def load_dataset(dataset):
    df=pd.read_csv(dataset)
    return df

def remove_non_ascii(text): 
    return ''.join(i for i in str(text) if ord(i)<128)

# remove urls
def remove_url(twitt):
    cleaned_twitt=re.sub(r'http\S+', '', twitt)
    return cleaned_twitt

# sentence decontraction
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# remove numbers and punctions and keep only letters from text
def keep_letters(text):
    "remove punctuations and downcase"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', text).lower()
    sentence = re.sub("\s\s+", " ", sentence)
    sentence = sentence.split(" ")
    sentence = " ".join(sentence)
    return sentence

def Punctuation(sentence): 
  
    # punctuation marks 
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  
    # traverse the given string and if any punctuation 
    # marks occur replace it with null 
    for x in sentence.lower(): 
        if x in punctuations: 
            sentence = sentence.replace(x, "") 
  
    return sentence

# spelling correction using TextBlob package
def spelling_correction(tweet):
    tweet_blob = TextBlob(tweet)
    corrected_tweet = tweet_blob.correct()
    return ' '.join(corrected_tweet.words)

# remove all digits
def remove_digits(text):
    result = ''.join([i for i in text if not i.isdigit()])
    return result

def remove_stop_words(text):
    # tokenize text
    sentence = text.split(" ")
    # define stop words
    # from nltk list:
    STOP_WORDS = nltk.corpus.stopwords.words('english')
    # add words to STOP_WORDS list manually:
    more_stop_words=['like','us','amp', 'pm', 'rt', 'nc', 'rd', 'u', 'c', 'mh', 'mp', 'im']
    for word in more_stop_words:
        STOP_WORDS.append(word)
    # remove words from STOP_WORDS list manually:
    less_stop_words = ['not', 'no']
    for word in less_stop_words:  # iterating on a copy since removing will mess things up
        if word in STOP_WORDS:
            STOP_WORDS.remove(word)
    # remove stop words from input text
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)        
    sentence = " ".join(sentence)
    return sentence

# lemmatization - extracting grammatical root of words (playing -> play)
def lemmatization(tweet):
    tweet_list=tweet.split(" ")
    lem = WordNetLemmatizer()
    normalized_tweet = []
    for word in tweet_list:
        normalized_text = lem.lemmatize(word,'v')
        normalized_tweet.append(normalized_text)
    normalized_tweet=' '.join(normalized_tweet)
    return normalized_tweet

# NER (Named Entity Recognition) function with keeping nouns, verbs and adjectives
def NER_process(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    # find the nouns:
    is_noun = lambda pos: pos[:2] == 'NN'
    nouns = [word for (word, pos) in sent if is_noun(pos)]
    # find the adjectives:
    is_adj = lambda pos: pos[:2] == 'JJ'
    adjs = [word for (word, pos) in sent if is_adj(pos)]
    # find the verbs:
    is_verb = lambda pos: pos[:2] == 'VB'
    verbs = [word for (word, pos) in sent if is_verb(pos)]
    # find the adverbs
    is_adverb = lambda pos: pos[:2] == 'RB'
    adverbs = [word for (word, pos) in sent if is_adverb(pos)]
    # append into a single list
    for adj in adjs:
        nouns.append(adj)
    for verb in verbs:
        nouns.append(verb)
    for adv in adverbs:
        nouns.append(adv)
    # return as a string containing all the words:
    final=str()
    for el in nouns:
        final=final+' '+el
    return final

# parsing (tokenization)
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent

def word_cloud(text, title):

    wordcloud = WordCloud(max_font_size=50, max_words=300, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig('figures/word_cloud_'+title+'.png')
    plt.show()

def word_frequency_barplot(corpus, title, nb_top_words=30):
    '''
    function to plot most frequent words' frequency from a corpus
    '''
    # flatten corpus to get a single list containing all the words
    corpus_flattened=[]
    for el in corpus:
        corpus_flattened.append(el[0])
    
    # import Counter function
    from collections import Counter 
    
    counter=Counter(corpus_flattened)
    most_occur=counter.most_common(nb_top_words)
    
    # get words and their occurences
    most_frequent_words=list(zip(*most_occur))[0]
    occurences=list(zip(*most_occur))[1]
    most_frequent_words=list(most_frequent_words)
    occurences=list(occurences)
    
    # plot 
    fig, ax = plt.subplots(1,1,figsize=(20,5))
    sns.barplot(most_frequent_words, occurences, color="green", ax=ax)
    ax.set_xticks(list(range(nb_top_words)))
    ax.set_xticklabels(most_frequent_words, fontsize=14, rotation=60)
    plt.tight_layout()
    plt.savefig('figures/word_frequencies_'+title+'.png')
    plt.show()
    return ax

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

def word_length_distribution(corpus):
    lengths = [len(i) for i in corpus]
    fig, ax = plt.subplots(figsize=(15,6))
    ax.set_title("Distribution of number of words", fontsize=16)
    ax.set_xlabel("Number of words", fontsize=14)
    ax.set_ylabel("Frequency", fontsize=14)
    ax.tick_params(labelsize=14)
    sns.distplot(lengths, bins=np.array(lengths).max(), ax=ax)
    plt.show()

# run all cleaning functions at once
def clean_tweet(tweet):
    
    cleaned_tweet=remove_non_ascii(tweet)
    cleaned_tweet=remove_url(tweet)
    cleaned_tweet=decontracted(cleaned_tweet)
    cleaned_tweet=keep_letters(cleaned_tweet)
    cleaned_tweet=remove_digits(cleaned_tweet)
    cleaned_tweet=remove_stop_words(cleaned_tweet)
    cleaned_tweet=NER_process(cleaned_tweet)
    
    return str(cleaned_tweet)


########################################## EXECUTION ###########################################

# load dataset
df_train=load_dataset('data/train.csv')
df_test=load_dataset('data/test.csv')
# define copy of datasets
df_train_cleaned=df_train[['id','text','target']]
df_test_cleaned=df_test[['id','text']]

df_train_cleaned=df_train_cleaned[(df_train_cleaned['target']==0) | (df_train_cleaned['target']==1)]
df_train_cleaned.drop_duplicates(subset=['text','target'],inplace=True)
df_test_cleaned.drop_duplicates(subset=['text'],inplace=True)
df_train_cleaned.dropna(subset=['text'], inplace=True)
df_test_cleaned.dropna(subset=['text'], inplace=True)

print('shape train set: ', df_train_cleaned.shape)
print('shape test set: ', df_test_cleaned.shape)

# testing clean_tweet function
test=df_train_cleaned.at[577,'text']
print('before cleaning:', test)
print('after cleaning:', clean_tweet(test))

# create new column containing cleaned twitt
df_test_cleaned['cleaned_text']=df_test_cleaned['text'].progress_apply(lambda x: clean_tweet(x))
df_train_cleaned['cleaned_text']=df_train_cleaned['text'].progress_apply(lambda x: clean_tweet(x))
df_test_cleaned.to_csv('data/test_cleaned.csv',index=False)
df_train_cleaned.to_csv('data/train_cleaned.csv',index=False)

# EDA
df_train_processed=load_dataset('data/train_cleaned.csv')
df_train_processed.dropna(inplace=True)
df_test_processed=load_dataset('data/test_cleaned.csv')
df_test_processed.dropna(inplace=True)

print('shape train set: ', df_train_processed.shape)
print('shape test set: ', df_test_processed.shape)

# disaster tweets
disaster_tweets=df_train_processed[df_train_processed['target']==1]
text_disaster=disaster_tweets.loc[:,'cleaned_text'].tolist()
text_disaster=','.join(text_disaster)
word_cloud(text_disaster, 'disaster_tweets')

# disaster tweets
no_disaster_tweets=df_train_processed[df_train_processed['target']==0]
text_no_disaster=no_disaster_tweets.loc[:,'cleaned_text'].tolist()
print(text_no_disaster[3951])
text_no_disaster=','.join(text_no_disaster)
word_cloud(text_no_disaster, 'NO_disaster_tweets')

# Most frequent words
text_disaster_list=build_corpus(disaster_tweets.loc[:,'cleaned_text'])
text_no_disaster_list=build_corpus(no_disaster_tweets.loc[:,'cleaned_text'])
word_frequency_barplot(text_disaster_list, 'disaster_tweets', 40)
word_frequency_barplot(text_no_disaster_list, 'NO_disaster_tweets', 40)

all_tweets=build_corpus(df_train_processed.loc[:,'cleaned_text'])
word_length_distribution(all_tweets)