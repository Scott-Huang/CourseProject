import numpy as np
from text_parser import get_parsed_data
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

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
def topic_model(model='lda', num_topic=15):
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

def topic_diff(topic1, topic2, p=3.0):
    diff = topic1 - topic2
    return np.power((diff ** p).sum(), 1/p)

# @TODO evaluate topic models by comparing their topic diversification 
#       this require a word2vec model to get word similarity
#       and I might skip this part and directly test their performance on paragraph segregation