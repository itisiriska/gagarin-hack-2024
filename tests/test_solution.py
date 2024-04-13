import pytest

import final_solution

ner = final_solution.solution.init_ner()
sentiment = final_solution.solution.init_sentiment()


def test_empty():
    """If score_texts do not pass this test, it is fine"""

    assert not bool(final_solution.solution.score_texts([], ner=ner, sentiment=sentiment))

    # nothing was found
    nothing = final_solution.solution.score_texts([""], ner=ner, sentiment=sentiment)
    assert nothing == [[tuple()]]


def test_one_message():
    """Format of answers is important"""
    messages = ["Сбер, он и в Африке Сбер"]
    correct_scores = [[(150, 3.0)]]
    
    assert final_solution.solution.score_texts(messages, ner=ner, sentiment=sentiment) == correct_scores


def test_two_entities_one_message():
    """Order of companies inside one message is not important"""
    messages = ["Сбер, он и в Африке. Тинькофф, он и в Африке Тинькофф"]
    correct_scores = [[(150, 3.0), (225, 3.0)]]
    
    scores = final_solution.solution.score_texts(messages, ner=ner, sentiment=sentiment)

    assert [set(s) == set(cs) for s, cs in zip(scores, correct_scores)]


def test_two_entities_two_messages():
    """"""
    messages = ["Сбер, он и в Африке Сбер", "Тинькофф, он и в Африке Тинькофф"]
    correct_scores = [[(150, 3.0)], [(225, 3.0)]]

    assert final_solution.solution.score_texts(messages, ner=ner, sentiment=sentiment) == correct_scores


def test_large_sequence(N = 10 ** 3):
    """No matter how large N is, score_texts function should work"""
    message = "Сбер, он и в Африке Сбер"
    correct_score = [(150, 3.0)]

    assert final_solution.solution.score_texts([message] * N, ner=ner, sentiment=sentiment) == [correct_score] * N
