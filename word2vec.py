import multiprocessing
from gensim.models import Word2Vec
from text_parser import get_parsed_data


cores = multiprocessing.cpu_count()

def train_model():
    data = list(get_parsed_data().values())
    w2v_model = Word2Vec(min_count=1,vector_size=64,negative=5,workers=cores-2)
    w2v_model.build_vocab(data)
    w2v_model.train(data, total_examples=w2v_model.corpus_count, 
                    epochs=80, report_delay=60)
    w2v_model.save('word2vec.model')

def get_model():
    return Word2Vec.load('word2vec.model')