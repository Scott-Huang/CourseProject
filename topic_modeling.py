"""
This program handles functions that measure differences of sentences.
There are many topic and bert models here too, mainly used for calculating sentence differences.
"""

import numpy as np
from text_parser import get_parsed_data, tokenize
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from functools import lru_cache

import warnings
warnings.filterwarnings('ignore')

data = get_parsed_data()

# usually bi-gram model will not have much different with unigram model
vectorizer = CountVectorizer(min_df=1, max_df=0.9, stop_words='english', lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
data_vectorized = vectorizer.fit_transform(list(data.values()))

def get_vec_model():
    return vectorizer
def get_vec_data():
    return data_vectorized

TOPIC_MODELS = ['lda', 'nmf', 'lsi']
@lru_cache(maxsize=4)
def topic_model(model='lda', num_topic=15):
    topic_model.current_model_name = None
    if model not in TOPIC_MODELS:
        raise Exception("Model not supported.")
    if model == 'lda':
        lda = LatentDirichletAllocation(num_topic, max_iter=25, learning_method='online',verbose=True)
        lda.fit_transform(data_vectorized)
        return lda
    elif model == 'nmf':
        nmf = NMF(num_topic)
        nmf.fit_transform(data_vectorized)
        return nmf
    else:
        lsi = TruncatedSVD(num_topic)
        lsi.fit_transform(data_vectorized)
        return lsi

def selected_topics(model, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])

def predict_topic(model, sentence:str):
    sentence = vectorizer.transform([sentence.lower()]).toarray()
    return model.transform(sentence)

def topic_diff(topic1, topic2, p=3.0, num_topic=15):
    diff = topic1 - topic2
    diff = diff.reshape(-1,num_topic)
    return np.power(np.abs(diff ** p).sum(axis=1), 1/p)

def baseline_sent_diff(sents,model):
    model = topic_model(model)
    sents_topic = np.array([predict_topic(model,tokenize(sent)) 
                            for sent in sents])
    sents_diff = topic_diff(sents_topic[:-1],sents_topic[1:])
    return sents_diff

# @TODO evaluate topic models by comparing their topic diversification 
#       this require a word2vec model to get word similarity
#       and I might skip this part and directly test their performance on paragraph segregation

import torch
from torch.nn import functional as F
from sentence_transformers import SentenceTransformer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# list of models
# 'all-MiniLM-L6-v2'
# 'gsarti/scibert-nli'
sentence_transformer = SentenceTransformer('gsarti/scibert-nli', device=device)

def text2vec(sentences):
    with torch.no_grad():
        return sentence_transformer.encode(sentences, convert_to_tensor=True)

def bert_sent_diff(sents):
    sents_vec = text2vec(sents)
    sents_diff = F.cosine_similarity(sents_vec[:-1],sents_vec[1:])
    return sents_diff.cpu()
