#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[8]:

# Import Python libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
from os import path as os_path
import time
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
from langdetect import detect
import nltk
nltk.download('popular')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import KBinsDiscretizer
import pickle
import en_core_web_sm

# Library for PEP8 standard
# from nbpep8.nbpep8 import pep8
from PIL import Image
image = Image.open('stack_overflow_logo.png')

st.image(image)


def remove_pos(nlp, x, pos_list):
    doc = nlp(x)
    list_text_row = []
    for token in doc:
        if(token.pos_ in pos_list):
            list_text_row.append(token.text)
    join_text_row = " ".join(list_text_row)
    join_text_row = join_text_row.lower().replace("c #", "c#")
    return join_text_row

def text_cleaner(x, nlp, pos_list, lang="english"):
    """Function allowing to carry out the preprossessing on the textual data. 
        It allows you to remove extra spaces, unicode characters, 
        English contractions, links, punctuation and numbers.
        
        The re library for using regular expressions must be loaded beforehand.
        The SpaCy and NLTK librairies must be loaded too. 

    Parameters
    ----------------------------------------
    x : string
        Sequence of characters to modify.
    ----------------------------------------
    """
    # Remove POS not in "NOUN", "PROPN"
    x = remove_pos(nlp, x, pos_list)
    # Case normalization
    x = x.lower()
    # Remove unicode characters
    x = x.encode("ascii", "ignore").decode()
    # Remove English contractions
    x = re.sub("\'\w+", '', x)
    # Remove ponctuation but not # (for C# for example)
    x = re.sub('[^\\w\\s#]', '', x)
    # Remove links
    x = re.sub(r'http*\S+', '', x)
    # Remove numbers
    x = re.sub(r'\w*\d+\w*', '', x)
    # Remove extra spaces
    x = re.sub('\s+', ' ', x)
        
    # Tokenization
    x = nltk.tokenize.word_tokenize(x)
    # List of stop words in select language from NLTK
    stop_words = stopwords.words(lang)
    # Remove stop words
    x = [word for word in x if word not in stop_words 
         and len(word)>2]
    # Lemmatizer
    wn = nltk.WordNetLemmatizer()
    x = [wn.lemmatize(word) for word in x]
    
    # Return cleaned text
    return x






st.title('Tag Predictor')


#@st.cache

Project_path = os_path.abspath(os_path.split(__file__)[0])
print(Project_path)
model_path = Project_path + "/models/model.pickle"
vectorizer_path = Project_path + "/models/vectorizer.pickle"
multilabel_binarizer_path = Project_path + "/models/multilabel_binarizer.pickle"

with st.form("my_form", clear_on_submit = True):
    title = st.text_input("Title ?")
    body = st.text_area("Question ?")
    submitted = st.form_submit_button("Predict")


if submitted:
    with st.spinner('Wait for it...'):
        #nlp = spacy.load('en_core_web_sm')
        nlp = en_core_web_sm.load()
        pos_list = ["NOUN","PROPN"]
        fulldoc = body + title
        fulldoc = text_cleaner(fulldoc, nlp, pos_list,"english")
        #if fulldoc:
        #      st.write(fulldoc)
        #loader le vectorizer
        #st.write(fulldoc)
        #st.write(vectorizer_path)
        #st.write(model_path)
        vectorizer = pickle.load(open(vectorizer_path,'rb'))
        #if vectorizer:
        #      st.write("Vectorizer OK")
        X_tfidf = vectorizer.transform([fulldoc])
        #st.write(X_tfidf)
        #loader le model
        model = pickle.load(open(model_path,'rb'))
        #if model:
        #      st.write("Model OK")
        #      st.write(model)
        prediction = model.predict(X_tfidf)
        #st.write("on est la")
        multilabel_binarizer = pickle.load(open(multilabel_binarizer_path,'rb'))
        tags_predict = multilabel_binarizer.inverse_transform(prediction)
    st.success('Done!')   
    st.subheader(title)
    st.caption(body)
    st.write(tags_predict[0])
     
    
# In[ ]:




