import os
import re
import pickle

import emoji
import nltk
import typing as tp
import pandas as pd

from string import punctuation

from flashtext import KeywordProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


EntityScoreType = tp.Tuple[int, float]  # (entity_id, entity_score)
MessageResultType = tp.List[
    EntityScoreType
]  # list of entity scores,
#    for example, [(entity_id, entity_score) for entity_id, entity_score in entities_found]

NER_PATH = 'final_solution/ner_dict.pkl'
SENTIMENT_PATH = 'final_solution/sentiment.pkl'

nltk.download('punkt')


def init_ner():
    """Loads dict for prefix tree"""
    with open(NER_PATH, 'rb') as file:
        keyword_dict = pickle.load(file)

    ner = KeywordProcessor()
    ner.add_keywords_from_dict(keyword_dict)
    return ner


def init_sentiment():
    """Loads model for sentiment analysis"""
    if not os.path.exists(SENTIMENT_PATH):
        print(f'Please load sentiment model by the following link: https://disk.yandex.ru/d/iJfnLSAo59J7bA and put it to {SENTIMENT_PATH}')
        exit()
    
    with open(SENTIMENT_PATH, 'rb') as file:
        model = pickle.load(file)

    return model


def _predict(messages, ner, sentiment):
    text = messages['text']
    sentiments = []
    if len(messages['mention']) == 0:  # didn't found any companies
        return []
    if len(messages['mention']) == 1:  # found only one company
        return [(messages['mention'][0], sentiment.predict([text])[0])]

    if emoji.emoji_count(text) > 0:  # found multiple companies, message contains emojis 
        texts = [
            s.strip() for s in re.split(
                r':[а-я_]+:', emoji.demojize(text, language='ru')
            ) if s.strip()
        ]
    else:  # tokenize by sentence endings
        texts = nltk.sent_tokenize(text)
        
    # find sentences where exactly every company is mentioned
    text_mention = [
        (txt, mention or None) for txt in texts
        for mention in ner.extract_keywords(txt)
    ]
    sentiments = [
        (tm[1], sentiment.predict([tm[0]])[0])
        for tm in text_mention if tm[1]
    ]
    return sentiments


def score_texts(
    messages: tp.Iterable[str], *args, **kwargs
) -> tp.Iterable[MessageResultType]:
    """
    Main function (see tests for more clarifications)
    Args:
        messages (tp.Iterable[str]): any iterable of strings (utf-8 encoded text messages)

    Returns:
        tp.Iterable[tp.Tuple[int, float]]: for any messages returns MessageResultType object
    -------
    Clarifications:
    >>> assert all([len(m) < 10 ** 11 for m in messages]) # all messages are shorter than 2048 characters
    """
    ner = kwargs['ner']
    sentiment = kwargs['sentiment']
    df = pd.DataFrame({'text': messages})
    df['mention'] = df['text'].apply(ner.extract_keywords).apply(lambda x: list(set(x)))
    df['predicts'] = df.apply(lambda x: _predict(x, ner, sentiment), axis=1)
    return df['predicts'].tolist()
