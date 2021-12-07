from keybert import KeyBERT
from topic_modeling import sentence_transformer
#from transformers import AutoModel
#model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
#kw_model = KeyBERT(model)
kw_model = KeyBERT(sentence_transformer)

def extract_keywords(paragraph, keyword_num=6, diversity=0.25, threshold=0.25):
    keys = kw_model.extract_keywords(paragraph, keyphrase_ngram_range=(1, 3), 
                            top_n=keyword_num, stop_words='english', 
                            use_mmr=True, diversity=diversity)
    return [key for key,p in keys if p >= threshold]

def extract_keywords_all(paragraphs, keyword_num=6, diversity=0.45, threshold=0.3):
    all_keys = kw_model.extract_keywords(paragraphs, keyphrase_ngram_range=(1, 3), 
                            top_n=keyword_num, stop_words='english', 
                            use_mmr=True, diversity=diversity)
    return [[key for key,p in keys if p >= threshold] for keys in all_keys]