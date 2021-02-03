"""Code to run and test your implemented functions.
You do not need to submit this file.
"""

import numpy as np
import os
import time

from lab3 import e_step, m_step, fit_hmm


TEST_CASES = ['seq_short', 'seq_long']


def wrap_test(func):
    def inner():
        func_name = func.__name__.replace('test_', '')
        try:
            func()
            print('{}: PASSED'.format(func_name))
        except Exception as e:
            print('{}: FAILED, reason: {} ***'.format(func_name, str(e)))
    return inner


#@wrap_test
def test_e_step():

    for test_case in TEST_CASES:
        npzfile = np.load('data/{}.npz'.format(test_case))
        x_list = list(npzfile['x'])
        n_states = npzfile['n_states']

        # Use groundtruth as theta_old
        pi = npzfile['pi']
        A = npzfile['A']
        phi = {'mu': npzfile['mu'], 'sigma': npzfile['sigma']}

        # Run algo
        gamma_list, xi_list = e_step(x_list, pi, A, phi)

        # Check gamma
        assert len(gamma_list) == len(npzfile['gamma_list']), \
            'Gamma list is of incorrect length'
        for g in range(len(gamma_list)):
            assert np.allclose(gamma_list, npzfile['gamma_list']), \
                'Gamma incorrect'


        # Check xi
        assert len(xi_list) == len(npzfile['xi_list']), \
            'Xi is of incorrect length'
        for g in range(len(xi_list)):
            assert np.allclose(xi_list, npzfile['xi_list']), \
                'Xi incorrect'


#@wrap_test
def test_m_step():
    for test_case in TEST_CASES:
        npzfile = np.load('data/{}.npz'.format(test_case))
        x_list = list(npzfile['x'])
        gamma_list = list(npzfile['gamma_list'])
        xi_list = list(npzfile['xi_list'])

        pi, A, phi = m_step(x_list, gamma_list, xi_list)

        assert np.allclose(pi, npzfile['test_m_step_pi']), 'pi is incorrect'
        assert np.allclose(A, npzfile['test_m_step_A']), 'A is incorrect'
        assert np.allclose(phi['mu'], npzfile['test_m_step_mu']), \
            'mu is incorrect'
        assert np.allclose(phi['sigma'], npzfile['test_m_step_sigma']), \
            'sigma is incorrect'


def run_fit_hmm():

    for test_case in TEST_CASES:

        print('Running on {}'.format(test_case))
        print('---------------------')

        # Load data
        npzfile = np.load('data/{}.npz'.format(test_case))
        x = list(npzfile['x'])
        n_states = npzfile['n_states']
        phi = {'mu': npzfile['mu'], 'sigma': npzfile['sigma']}

        # Print groundtruth HMM parameters
        np.set_printoptions()
        print('Loaded {} sequences, with average length = {}'.format(
            len(x), np.mean([len(xi) for xi in x])
        ))
        print('Groundtruth pi:\n', npzfile['pi'])
        print('Groundtruth A:\n', npzfile['A'])
        print('Groundtruth phi:\n', phi)

        # Runs your code and prints out your parameters
        pi, A, phi = fit_hmm(x, n_states=n_states)
        np.set_printoptions(precision=2, suppress=True)
        print('Your pi:\n', pi)
        print('Your A:\n', A)
        print('Your phi:\n', phi)

        # Save out predictions
        with open('pred/{}.npz'.format(test_case), 'wb') as fid:
            np.savez(fid, pi=pi, A=A,
                     mu=phi['mu'], sigma=phi['sigma'])

        print('\n')


if __name__ == '__main__':
    t1 = time.time()
    os.makedirs('pred', exist_ok=True)
    # test_e_step()
    # test_m_step()
    run_fit_hmm()
    print(time.time()-t1)