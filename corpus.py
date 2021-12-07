"""
This program handles data collection.
"""

import glob
import pysrt
import webvtt
import json

def get_file_name(file):
    return file[file.rfind('\\')+1:]

def get_corpus(corrected=True, stored=False):
    """Process all subtitle files in the data folder.
    
    Parameters:
        corrected: If true, it will process all files in 
            data/corrected_transcripts which are in .vtt format, 
            otherwise, it will process the data/transcripts folder
            containing .srt files.
        stored: If true, storing the processed data into data/corpus.json.
    """
    data = {}
    if corrected:
        for file in glob.glob('data/corrected_transcripts/*/*.vtt'):
            file_name = get_file_name(file)
            data[file_name] = ' '.join([caption.text for caption in webvtt.read(file)])

    else:
        for file in glob.glob('data/transcripts/*/*.srt'):
            if 'textanalytics' in file:
                subfix = 'tm-' + file_name
            else:
                subfix = 'tr-' + file_name
            file_name = subfix + get_file_name(file)
            data[file_name] = ' '.join([caption.text.replace('\n', ' ').replace('[SOUND]','').replace('[MUSIC]','')
                                        for caption in pysrt.open(file)]).strip()

    if stored:
        with open('data/corpus.json', 'w+') as file:
            json.dump(data, file)

    return data
