# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 22:46:15 2021

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 08:30:25 2021

@author: admin
"""





#pip install streamlit



import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


ok=pd.read_excel("D:/5th Sem/Project/Copy of User Details_Faf.xlsx")



ok['essay']=ok[['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9']].agg(' '.join, axis=1)

ok.drop(['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','Name'],axis=1,inplace=True)



corpus_df = ok.copy(deep=True)




#regex
import re
# import natural language toolkit
import nltk
#beautiful soup
from bs4 import BeautifulSoup
#string for punctuation
import string
#stop word list
from nltk.corpus import stopwords
#import tokenizer
from nltk.tokenize import RegexpTokenizer
#import Lemmatizer
from nltk.stem import WordNetLemmatizer
#import stemmer
from nltk.stem.porter import PorterStemmer
#import html parser just in case BS4 doesn't work
import html.parser





corpus_df['corpus'] = ok[['age', 'status', 'sex', 'orientation', 'body_type', 'diet', 'drinks',
       'drugs', 'education', 'ethnicity', 'height', 'income', 'job',
       'offspring', 'pets', 'religion', 'sign', 'smokes', 'speaks', 'essay']].astype(str).agg(' '.join, axis=1)
corpus_df = corpus_df.astype(str)

# replaced \n
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('\n', ' '))

# replace all nan's and removed apostrophe
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('nan', ' '))
#removed apostrophe
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("\'", ""))
#remove dashes
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("-'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("--'", ""))
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("='", ""))
#remove forward slash
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("/", ""))
#remove periods
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(".", " "))

#remove colon
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(":", " "))

# remove comma
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(",", " "))

# remove left parentheses
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("(", " "))

#remove right parentheses
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(")", " "))

#remove question marks
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("?", " "))

#remove ! mark
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace("!", " "))

#remove semicolon marks
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace(";", " "))
# remove quotation marks
corpus_df['corpus'] = corpus_df['corpus'].map(lambda x: x.replace('"', " "))



# remove numbers
corpus_df['corpus'] = corpus_df['corpus'].str.replace('\d+', '')

corpus_list = corpus_df['corpus']

import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
     return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
 
nltk.download('wordnet')
corpus_df['processed_text'] = corpus_df['corpus'].apply(lemmatize_text)



#stop words
nltk.download('stopwords')
stops = set(stopwords.words("english"))   
corpus_df['processed_text'] = corpus_df['processed_text'].apply(lambda x: [item for item in x if item not in stops])





#tf - idf vectorization

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as its own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

tfidf = TfidfVectorizer(stop_words = "english", ngram_range = (1,3), max_df=0.8, min_df=0.2) 

corpus_tfidf = tfidf.fit(corpus_list)

corpus_2d = pd.DataFrame(tfidf.transform(corpus_list).todense(),
                   columns = tfidf.get_feature_names(),)

tfidf_vec = tfidf.fit_transform(corpus_list)

corpus_2d.head()

corpus_mat_sparse = csr_matrix(corpus_2d.values)

pd.set_option('display.max_columns',25)
pd.set_option('expand_frame_repr', False)




#Model specification
from sklearn.neighbors import NearestNeighbors

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(corpus_mat_sparse)


##recom algo

#recommendation algorithm (cosine)

def rec(query_index):
  distances, indices = model_knn.kneighbors(corpus_2d.iloc[query_index, :].values.reshape(1, -1), n_neighbors = 21)

  for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}: \n'.format(corpus_2d.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2} \n '.format(i, corpus_2d.index[indices.flatten()[i]],distances.flatten()[i]))
  for i in indices:
    print(okc.loc[i,:])


#getting recom for 15
rec(15)


'''
import pickle


#saving model to disk
pickle.dump(rec, open("rec_sys.pkl","wb"))


#loading model
reco = pickle.load(open("rec_sys.pkl","rb"))
print(reco(11))
'''



