"""Code to run and test your implemented functions.
You do not need to submit this file.
"""

import json
import os

from factor import load_factor_list_from_json, to_factor
from lab1 import *

STR_OUTPUT_MISMATCH = 'Output does not match.'
GRAPH_DATA = ['graph_small', 'graph_large']  # List of test graphs


def wrap_test(func):
    def inner():
        func_name = func.__name__.replace('test_', '')
        try:
            func()
            print('{}: PASSED'.format(func_name))
        except Exception as e:
            print('{}: FAILED, reason: {} ***'.format(func_name, str(e)))
    return inner


def load_test_cases(graph, test_function):
    """Load test cases for json file"""
    fname = os.path.join('data', '{}_tests.json').format(graph)
    with open(fname, 'r') as fid:
        test_cases = json.load(fid)

    if test_function not in test_cases:
        return []

    test_cases = test_cases[test_function]
    for test_case in test_cases:
        for field in test_case:
            if field.endswith('_factor'):
                test_case[field] = to_factor(test_case[field])
            elif field.endswith('_factors'):
                test_case[field] = [to_factor(f) for f in test_case[field]]
            elif field == 'evidence' or field == 'max_decoding':
                new_dict = {int(k): v for (k, v) in test_case[field].items()}
                test_case[field] = new_dict

    return test_cases


#@wrap_test
def test_factor_product():
    # Test case 1
    # factorA contains P(X_0)
    factor0 = Factor(var=[0],
                     card=[2],
                     val=[0.8, 0.2])
    # factor1 contains P(X_1|X_0)
    factor1 = Factor(var=[0, 1],
                     card=[2, 2],
                     val=[0.4, 0.55, 0.6, 0.45])

    correct = Factor(var=[0, 1],
                     card=[2, 2],
                     val=[0.32, 0.11, 0.48, 0.09])

    output = factor_product(factor0, factor1)
    assert output == correct, STR_OUTPUT_MISMATCH

    # Test case 2: No variables in common
    # factorA contains P(X_0)
    factor0 = Factor(var=[0],
                     card=[2],
                     val=[0.8, 0.2])
    # factor1 contains P(X_1)
    factor1 = Factor(var=[1],
                     card=[2],
                     val=[0.4, 0.6])

    correct = Factor(var=[0, 1],
                     card=[2, 2],
                     val=[0.32, 0.08, 0.48, 0.12])

    output = factor_product(factor0, factor1)
    assert output == correct, STR_OUTPUT_MISMATCH

#@wrap_test
def test_factor_marginalize():
    factor = Factor(var=[2, 3],
                    card=[2, 3],
                    val=[0.4, 0.35, 0.6, 0.45, 0.0, 0.2])
    vars_to_marginalize_out = [2]
    # factor = Factor(var=[1,2,3],card = [2,2,2],val = [1,1,1,1,1,1,1,1])
    # vars_to_marginalize_out = [3]

    correct = Factor(var=[3],
                     card=[3],
                     val=[0.75, 1.05, 0.2])

    output = factor_marginalize(factor, vars_to_marginalize_out)
    assert output == correct, STR_OUTPUT_MISMATCH



def test_observe_evidence():
    factors = [Factor(var=[0, 1],
                      card=[2, 3],
                      val=[0.4, 0.35, 0.6, 0.45, 0.0, 0.2]),
               Factor(var=[1, 2],
                      card=[3, 2],
                      val=[0.1, 0.2, 0.3, 0.9, 0.8, 0.7])]
    evidence = {0: 1}

    correct = [Factor(var=[0, 1],
                      card=[2, 3],
                      val=[0.0, 0.35, 0.0, 0.45, 0.0, 0.2]),
               Factor(var=[1, 2],
                      card=[3, 2],
                      val=[0.1, 0.2, 0.3, 0.9, 0.8, 0.7])]

    output = observe_evidence(factors, evidence)  # Marginalize out X_2
    assert output == correct, STR_OUTPUT_MISMATCH


#@wrap_test
def test_compute_marginals_naive():
    for graph in GRAPH_DATA:
        factors = load_factor_list_from_json(
            os.path.join('data', '{}.json').format(graph))

        test_cases = load_test_cases(graph, 'compute_marginals_naive')
        for test_case in test_cases:
            output = compute_marginals_naive(test_case['var'],
                                             factors,
                                             test_case['evidence'])

            assert output == test_case['correct_factor'], STR_OUTPUT_MISMATCH


#@wrap_test
def test_compute_marginals_bp():
    for graph in GRAPH_DATA:
        factors = load_factor_list_from_json(
            os.path.join('data', '{}.json').format(graph))

        test_cases = load_test_cases(graph, 'compute_marginals_bp')
        for test_case in test_cases:
            output = compute_marginals_bp(test_case['var'],
                                          factors,
                                          test_case['evidence'])
            assert output == test_case['correct_factors'], STR_OUTPUT_MISMATCH


#@wrap_test
def test_factor_sum():
    # factorA contains phi(X_0)
    factor0 = Factor(var=[0],
                     card=[2],
                     val=[0.8, 0.2])
    # factor1 contains phi(X_1|X_0)
    factor1 = Factor(var=[0, 1],
                     card=[2, 2],
                     val=[0.4, 0.55, 0.6, 0.45])

    correct = Factor(var=[0, 1],
                     card=[2, 2],
                     val=[1.2, 0.75, 1.4, 0.65])

    output = factor_sum(factor0, factor1)
    assert output == correct, STR_OUTPUT_MISMATCH


#@wrap_test
def test_factor_max_marginalize():
    factor = Factor(var=[3, 4, 5],
                    card=[2, 2, 3],
                    val=[0.1, 0.9, 0.3, 0.5, 0.4, 0.4, 0.2, 0.1, 0.4, 0.2, 0.7, 0.1])
    assignment = factor.get_all_assignments()
    vars_to_marginalize_out = [3, 5]  # Marginalize out X3, X5, leaving X4

    correct = Factor(var=[4],
                     card=[2],
                     val=[0.9, 0.7],
                     val_argmax=[
                         {3: 1, 5: 0},  # When X4==0, the maximizing assignment is X_3=1, X_5=0
                         {3: 0, 5: 2},  # When X4==1, the maximizing assignment is X_3=0, X_5=2
                     ])

    output = factor_max_marginalize(factor, vars_to_marginalize_out)
    assert output == correct, STR_OUTPUT_MISMATCH


# @wrap_test
def test_map_eliminate():
    for graph in GRAPH_DATA:
        factors = load_factor_list_from_json(
            os.path.join('data', '{}.json').format(graph))

        test_cases = load_test_cases(graph, 'map_eliminate')
        for test_case in test_cases[1:]:
            output = map_eliminate(factors, test_case['evidence'])
            assert np.allclose(output[1], test_case['log_prob']), 'log_prob wrong'
            assert output[0] == test_case['max_decoding'], 'Max decoding wrong'


if __name__ == '__main__':
    test_factor_product()
    test_factor_marginalize()
    test_observe_evidence()
    test_compute_marginals_naive()
    test_compute_marginals_bp()
    test_factor_sum()
    test_factor_max_marginalize()
    test_map_eliminate()

