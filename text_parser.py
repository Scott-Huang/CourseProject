import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from corpus import get_corpus

punctuations = string.punctuation
stopwords = list(STOP_WORDS)
parser = English()

def tokenize(sentence):
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ]
    mytokens = " ".join([i for i in mytokens])
    return mytokens

def get_parsed_data():
    data = get_corpus()

    for title,text in data.items():
        data[title] = tokenize(text)

    return data