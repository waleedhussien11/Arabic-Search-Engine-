import re
from snowballstemmer import stemmer
import nltk
from nltk.corpus import stopwords
import math
import pandas as pd
from collections import defaultdict
from pyarabic import araby
import pickle

nltk.download('stopwords')
stop_words = set(stopwords.words('arabic'))
def extract_text(file):
    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def clean(text):
    text = re.sub(r"http\S+", " ", text) # remove urls
    text = re.sub(r"RT ", " ", text) # remove rt
    text = re.sub(r"@[\w]*", " ", text) # remove handles
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text) # remove special characters
    text = re.sub(r'\t', ' ', text) # remove tabs
    text = re.sub(r'\n', ' ', text) # remove line jump
    text = re.sub(r"\s+", " ", text) # remove extra white space
    accents = re.compile(r'[\u064b-\u0652\u0640]') # harakaat and tatweel (kashida) to remove

    arabic_punc= re.compile(r'[\u0621-\u063A\u0641-\u064A\d+]+') # Keep only Arabic letters/do not remove number
    text=' '.join(arabic_punc.findall(accents.sub('',text)))
    text = text.strip()
    return text
def remove_stopwords(sentence):
    terms = [word for word in sentence.split() if word.lower() not in stop_words]
    return " ".join(terms)




def normalize(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    return(text)

#specify that we want to stem arabic text
ar_stemmer = stemmer("arabic")
#define the stemming function
def stem(sentence):
    return " ".join([ar_stemmer.stemWord(i) for i in sentence.split()])

def preprocess(sentence):
    
    sentence = clean(sentence)
    sentence = remove_stopwords(sentence)
    sentence = normalize(sentence)
    sentence = stem(sentence)
    
    return sentence