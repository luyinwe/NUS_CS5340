import os
import json
import argparse
import numpy as np
from factor import Factor

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
ANSWER_DIR = os.path.join(DATA_DIR, 'ground-truth')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


def parse_json(json_file: str):
    with open(json_file, 'r') as f:
        factor = json.load(f)
    factor = Factor(var=factor['var'], card=factor['card'], val=factor['val'])
    return factor


def check_answers(case: int, tolerance_decimal: int = 1):
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    answer_file = os.path.join(ANSWER_DIR, '{}.json'.format(case))
    predictions = parse_json(json_file=prediction_file)
    answers = parse_json(json_file=answer_file)

    np.testing.assert_equal(actual=predictions.var, desired=answers.var)
    np.testing.assert_equal(actual=predictions.card, desired=answers.card)
    np.testing.assert_almost_equal(actual=predictions.val, desired=answers.val, decimal=tolerance_decimal)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=int, required=True)
    args = parser.parse_args()
    check_answers(case=args.case)


if __name__ == '__main__':
    main()

