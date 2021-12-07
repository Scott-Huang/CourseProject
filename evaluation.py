import torch
import json
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import completeness_score, adjusted_mutual_info_score
from sentence_transformers import SentenceTransformer
from paragraph_segmentation import baseline_segmentation, partition, sentence_spliter
from keyword_extraction import extract_keywords_all

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
phrase_transformer = SentenceTransformer('whaleloops/phrase-bert', device=device)

def get_annotated_answer():
    with open('data/annotated_data.json') as file:
        annotated_answer = json.loads(file.read())
    for annotated_data in annotated_answer:
        annotated_data['keywords'] = eval(annotated_data['keywords'])
        annotated_data['segmentation'] = eval(annotated_data['segmentation'])
    return annotated_answer

def evaluate_segmentation(model, log=False):
    total_score = 0
    for annotation in get_annotated_answer():
        text = annotation['text']
        pred_arg,_ = baseline_segmentation(text,model)
        test_arg = np.array(annotation['segmentation']) + 1
        pred_arg = partition(text,pred_arg)
        test_arg = partition(text,test_arg)
        score = completeness_score(test_arg,pred_arg)
        score += adjusted_mutual_info_score(test_arg,pred_arg)
        total_score += score ** 2
    total_score = np.sqrt(total_score)

    if log:
        print('The average score of segmentation is %.4f.' % total_score)
    return total_score

def get_paragraph(text,arg):
    sents = sentence_spliter(text)
    paras = np.split(sents, np.array(arg)+1)
    return [' '.join(para) for para in paras]

def get_phrase_embedding(phrases):
    return phrase_transformer.encode(phrases, convert_to_tensor=True)

def precision(annotation):
    paragraphs = get_paragraph(annotation['text'], annotation['segmentation'])
    pred_keys = extract_keywords_all(paragraphs, keyword_num=5, diversity=0.45, threshold=0.25)
    test_keys = annotation['keywords']

    total_score = 0
    for pred_key,test_key in zip(pred_keys, test_keys):
        pred_embs = get_phrase_embedding(pred_key)
        test_embs = get_phrase_embedding(test_key)
        score = 0
        for pred_emb in pred_embs:
            score += F.cosine_similarity(pred_emb,test_embs,dim=1).max().item()
        total_score += score / len(pred_embs)
    avg_score = total_score / len(test_keys)
    return avg_score

def recall(annotation):
    paragraphs = get_paragraph(annotation['text'], annotation['segmentation'])
    pred_keys = extract_keywords_all(paragraphs, keyword_num=5, diversity=0.45, threshold=0.25)
    test_keys = annotation['keywords']

    total_score = 0
    for pred_key,test_key in zip(pred_keys, test_keys):
        pred_embs = get_phrase_embedding(pred_key)
        test_embs = get_phrase_embedding(test_key)
        score = 0
        for test_emb in test_embs:
            score += F.cosine_similarity(test_emb,pred_embs,dim=1).max().item()
        total_score += score / len(pred_embs)
    avg_score = total_score / len(test_keys)
    return avg_score

def evaluate_keyword_extraction(log=False):
    p,r = 0,0
    annotations = get_annotated_answer()
    for annotation in annotations:
        p += precision(annotation)
        r += recall(annotation)
    p /= len(annotations)
    r /= len(annotations)

    if log:
        print('Precision: %.3f' % p)
        print('Recall: %.3f' % r)
    return p,r
