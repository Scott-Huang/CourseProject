import spacy
import numpy as np
from topic_modeling import predict_topic, topic_diff
from text_parser import tokenize

nlp = spacy.load("en_core_web_sm")

def sentence_spliter(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def baseline_segmentation(text, model, num_segmentation=0, std=1, min_segregation=5, max_segregation=20):
    sents = sentence_spliter(text)

    sents_topic = []
    for sent in sents:
        sent = tokenize(sent)
        topic = predict_topic(model,sent)
        sents_topic.append(topic)
    sents_topic = np.array(sents_topic)

    sents_diff = topic_diff(sents_topic[:-1],sents_topic[1:])

    if not num_segmentation:
        threshold = sents_diff.mean() + std * sents_diff.std()
        num_segmentation = np.count_nonzero(sents_diff > threshold)
    num_segmentation = np.min([num_segmentation, max_segregation])
    num_segmentation = np.max([num_segmentation, min_segregation])
    
    arg = np.sort(np.argpartition(sents_diff, -num_segmentation-1)[-num_segmentation-1:])
    segmented_sents = np.split(sents,arg)

    output = []
    for i in segmented_sents:
        output.append(' '.join(i))
    
    # return arg for evaluation
    return arg,output

# for comparing partitioned segmentation
def partition(sents, arg):
    if sents is str:
        sents = sentence_spliter(sents)
    sents = np.zeros(len(sents))

    category = 0
    prev = 0
    for i in arg:
        sents[prev:i] = category
        category += 1
        prev = i
    sents[prev:] = category

    return sents
