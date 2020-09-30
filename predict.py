
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

import re
import string
import pickle
import time
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# eng 

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# indo

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# stoword indonesia

def stopword():

    stopword_indo_list = pd.read_csv('setting/stopword_ind.csv')
    stopword_indo_list = set(stopword_indo_list.text.values)

    # stopword  english

    stopword_eng_list = set(stopwords.words('english'))
    stopword_eng_list

    return stopword_indo_list, stopword_eng_list


def load_models():
    
    # Load the vectoriser.

    file = open('vectoriser-ngram-(1,2).pickle', 'rb')
    vectoriser = pickle.load(file)
    file.close()

    # Load the LR Model.

    file = open('Sentiment-BNB.pickle', 'rb')
    LRmodel = pickle.load(file)
    file.close()
    
    return vectoriser, LRmodel


def predict(vectoriser, model, text):

    # Predict the sentiment

    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment.
    data = []
    for text, pred in zip(text, sentiment):
        data.append((text,pred))
        
    # Convert the list into a Pandas DataFrame.

    df = pd.DataFrame(data, columns = ['text','sentiment'])
    df = df.replace([0,4,2], ["Negative","Positive",'NEUTRAL'])

    return df



def preprocess(textdata):

    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    stopword_indo_list, stopword_eng_list = stopword()
    
    # Defining regex patterns.
    urlPattern        = 'https?://\S+|www\.\S+'
    userPattern       = '@[\w]*'
    alphaPattern      = '[^a-zA-Z]'
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
   
    
    for text in textdata:

        text = str(text).lower()
        
        # Replace all URls with 'URL'
        text = re.sub(urlPattern,'',text)

        text = re.sub(userPattern,'', text)       
        # Replace all non alphabets.
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        # Replace all non alphabets.
        text = re.sub(alphaPattern, ' ', text)
        # delet space
        text = re.sub(' +', ' ', text)

        tweetwords = ''
        for word in text.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if word not in stopword_eng_list and word not in stopword_indo_list:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                word = stemmer.stem(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)
        
    return processedText


