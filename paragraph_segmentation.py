"""
This program handles all paragraph segmentation task.
"""

import spacy
import numpy as np
from topic_modeling import baseline_sent_diff, bert_sent_diff

nlp = spacy.load("en_core_web_sm")

def sentence_spliter(text):
    """Split the whole text read from file into sentences."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def refine(arg, threshold=3):
    """If the splited paragraphs containing sentences less than the threshold, 
    merge it with its previous paragraph.
    """
    diff = arg[1:] - arg[:-1]
    diff = np.insert(diff, 0, threshold+1)
    return arg[diff >= threshold]

def baseline_segmentation(text, model, num_segmentation=0,
                          std=.95, min_segmentation=2, max_segmentation=6):
    """A baseline approach that segments sentences into paragraphs
    based on their differences of adjacent sentences. 
    """
    sents = sentence_spliter(text)
    if model == 'bert':
        sents_diff = bert_sent_diff(sents)
    else:
        sents_diff = baseline_sent_diff(sents,model)

    if not num_segmentation:
        threshold = sents_diff.mean() + std * sents_diff.std()
        num_segmentation = np.count_nonzero(sents_diff > threshold)
    num_segmentation = np.max([np.min([num_segmentation, max_segmentation]), min_segmentation])
    
    arg = np.sort(np.argpartition(sents_diff, -num_segmentation-1)[-num_segmentation-1:])
    arg = refine(arg) + 1
    segmented_sents = np.split(sents,arg)

    output = []
    for i in segmented_sents:
        output.append(' '.join(i).strip())
    
    # Return arg for evaluation.
    return arg,output


def alpha_segmentation():
    """
    @TODO Implement an improved segmentation algorithm 
        that utilizes differences among all sentences 
        instead of only adjacent sentences.
    """
    pass

def end_to_end_segmentation():
    """
    @TODO Implement an end-to-end model that output keywords and paragraph
        segmentation at the same time, which requires sufficient data for
        supervised learning and some complicated model.
    """
    pass


def partition(sents:list or str, arg):
    """For comparing partitioned segmentation."""
    if isinstance(sents, str):
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
