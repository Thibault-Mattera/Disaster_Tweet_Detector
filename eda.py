# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis

"""

########################################## DEPENDENCIES ###########################################


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 13, 9

############################################ FUNCTIONS ############################################

def load_dataset(dataset):
    df=pd.read_csv(dataset)
    return df

def plot_bar_x(df,xlabel, feature):
    # this is for plotting purpose
    #lab=list(df[xlabel].unique())
    index = np.arange(len(xlabel))
    df.plot.bar(color={"disaster": "tomato", "no disaster": "lightgreen"})
    plt.grid()
    plt.xlabel(feature, fontsize=15)
    plt.ylabel('count', fontsize=15)
    plt.xticks(index, xlabel, fontsize=15, rotation=45, horizontalalignment='right')
    plt.yticks(fontsize=15, rotation=0, horizontalalignment='right')
    plt.title('Tweet ' + feature + ' analysis', fontsize=15)
    plt.legend(prop={'size': 14})
    plt.tight_layout()
    plt.savefig('figures/'+feature +'_analysis.png')
    plt.show()

############################################ EXECUTION ############################################

df_train=load_dataset('data/train.csv')
df_train.dropna(subset=['text','target'],inplace=True)
df_train['location'].fillna('-', inplace=True)
df_train['keyword'].fillna('-', inplace=True)
df_train['keyword']=df_train['keyword'].apply(lambda x: str(x).replace('%20', ' '))

top_keywords=df_train[df_train['keyword']!='-']['keyword'].value_counts().index.tolist()
top_keywords=top_keywords[:20]
print('top keywords', top_keywords)

df_train_top_keywords=df_train[df_train['keyword'].isin(top_keywords)]
df_train_top_keywords['target'].replace({0:'no disaster', 1:'disaster'}, inplace=True)
cross_tab=pd.crosstab(df_train_top_keywords['keyword'], df_train_top_keywords['target'])
cross_tab.sort_values(by='no disaster', ascending=False, inplace=True)
plot_bar_x(cross_tab,top_keywords, 'keywords')


top_locations=df_train[df_train['location']!='-']['location'].value_counts().index.tolist()
top_locations=top_locations[:22]
print('top_locations', top_locations)

df_train_top_locations=df_train[df_train['location'].isin(top_locations)]
df_train_top_locations['location'].replace('United States', 'USA', inplace=True)
df_train_top_locations['location'].replace('United Kingdom', 'UK', inplace=True)
df_train_top_locations['location'].replace('California, USA', 'California', inplace=True)
df_train_top_locations['location'].replace('New York, NY', 'New York', inplace=True)
df_train_top_locations['target'].replace({0:'no disaster', 1:'disaster'}, inplace=True)
cross_tab=pd.crosstab(df_train_top_locations['location'], df_train_top_locations['target'])
cross_tab.sort_values(by='disaster', ascending=False, inplace=True)
top_locations=cross_tab.index.tolist()
print(cross_tab)
plot_bar_x(cross_tab,top_locations, 'locations')
