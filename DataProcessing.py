import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

from bs4 import BeautifulSoup
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from keras.preprocessing import sequence
from sklearn.preprocessing import StandardScaler

class get_data:
    def __init__(self):
        return None

    def word2vec (self):

        embedding_size = 100
        maxlen = 130

        stop_words = set(stopwords.words('english'))

        def clean(sentence):
            # 1. Remove HTML
            review_text = BeautifulSoup(sentence, "lxml").get_text()
            # 2. Remove non-letters
            letters_only = re.compile(r'[^A-Za-z\s]').sub(" ",review_text)
            # 3. Convert to lower case
            lowercase_letters = letters_only.lower()
            return lowercase_letters

        def lemmatize(tokens):
            # 1. Lemmatize
            tokens = list(map(WordNetLemmatizer().lemmatize, tokens))
            lemmatized_tokens = list(map(lambda x: WordNetLemmatizer().lemmatize(x, "v"), tokens))
            # 2. Remove stop words
            meaningful_words = list(filter(lambda x: not x in stop_words, lemmatized_tokens))
            return meaningful_words

        def preprocess(sentences):
            sentences_processed = []
            for sentence in sentences:
                # 1. Clean text
                sentence = clean(sentence)
                # 2. Split into individual words
                tokens = word_tokenize(sentence)
                # 3. Lemmatize and remove stop words
                lemmas = lemmatize(tokens)
                sentences_processed.append(lemmas)
            return sentences_processed

        def trainModel(tokens, embedding_size): #Train skip-gram word2vec model
            model = Word2Vec(tokens, size=embedding_size, min_count=3,sg=1,hs=1)
            model.save('./Word2Vec_model')
            return None

        def words2Array (lineList): #Use word2vec model to convert each word in each sentence to vector
            model = Word2Vec.load('./Word2Vec_model')
            lineArray = []
            for line in lineList:
                wordArray =[]
                for word in line:
                    try:
                        vec_norm = model[word]
                        wordArray.append(vec_norm)
                    except KeyError:
                        continue
                lineArray.append(wordArray)
            return lineArray


        #========================================================================================================
        #Read data and preprocess them
        additionalData = pd.read_csv('./useful_data/additionalData.csv', encoding="latin-1")
        additionalData = additionalData[additionalData.label != 'unsup']    #Delete samples with 'unsup'
        additional_label = additionalData['label'].map({'pos': 1, 'neg': 0})
        additional_label = np.array(additional_label.values).flatten()
        additional_sentences = preprocess(additionalData['review'])

        unlabeledTrainData = pd.read_csv('./useful_data/unlabeledTrainData.tsv', delimiter='\t',error_bad_lines=False)
        unlabeledTrain_sentences = preprocess(unlabeledTrainData['review'])

        labeledTrainData = pd.read_csv('./useful_data/labeledTrainData.tsv', delimiter='\t')
        labeledTrain_label = np.array(labeledTrainData['sentiment'].values).flatten()
        labeledTrain_sentences = preprocess(labeledTrainData['review'])

        labeledTestData = pd.read_csv('./useful_data/labeledTestData.tsv', delimiter='\t')
        labeledTest_label = np.array(labeledTestData['sentiment'].values).flatten()
        labeledTest_sentences = preprocess(labeledTestData['review'])

        #==========================================================================================================
        #Train word2vec model
        word2vecTrainData = additional_sentences + unlabeledTrain_sentences + labeledTrain_sentences
        trainModel(word2vecTrainData, embedding_size)

        #===========================================================================================================
        #prepare the data for training language model
        X_train_raw = additional_sentences + labeledTrain_sentences   #Data for training language model
        #print(max(len(i) for i in X_train_raw))
        #print(sum(len(i) for i in X_train_raw)/len(X_train_raw)) #The average length is 121
        X_train = sequence.pad_sequences(words2Array(X_train_raw), dtype='float32',maxlen=maxlen)   #Make each sequence the same length
        Y_train = np.append(additional_label, labeledTrain_label)

        X_test_raw = labeledTest_sentences
        X_test = sequence.pad_sequences(words2Array(X_test_raw), dtype='float32',maxlen=maxlen)
        Y_test = labeledTest_label

        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)
        scaler = StandardScaler()   #Standardize the input data
        X_train = scaler.fit_transform(X_train.reshape(-1,1))
        X_train = X_train.reshape(-1,maxlen,embedding_size)
        X_test = scaler.transform(X_test.reshape(-1,1))
        X_test = X_test.reshape(-1,maxlen,embedding_size)
        X_val = scaler.transform(X_val.reshape(-1,1))
        X_val = X_val.reshape(-1,maxlen,embedding_size)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test
