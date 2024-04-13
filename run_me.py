import json
import pathlib
import typing as tp

import final_solution
# import final_solution.solution_stupid


PATH_TO_TEST_DATA = pathlib.Path("data") / "test_texts.json"
PATH_TO_OUTPUT_DATA = pathlib.Path("results") / "output_scores.json"


ner = final_solution.solution.init_ner()
sentiment = final_solution.solution.init_sentiment()


def load_data(path: pathlib.PosixPath = PATH_TO_TEST_DATA) -> tp.List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def save_data(data, path: pathlib.PosixPath = PATH_TO_OUTPUT_DATA):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)


def main():
    texts = load_data()
    scores = final_solution.solution.score_texts(
        texts, ner=ner, sentiment=sentiment
    )
    casted_scores = [  # from numpy int64 to serializeable int
        [tuple(map(int, tup)) for tup in sublst]
        for sublst in scores
    ]
    save_data(casted_scores)


if __name__ == '__main__':
    main()
