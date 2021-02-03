""" CS5340 Lab 4 Part 1: Importance Sampling
See accompanying PDF for instructions.

Name: <Your Name here>
Email: <username>@u.nus.edu
Student ID: A0123456X
"""

import os
import json
import numpy as np
import networkx as nx
from factor_utils import factor_evidence, factor_product, assignment_to_index
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """
def after_evidence(evidence, factors):
    for index in factors:
        factors[index] = factor_evidence(factors[index], evidence)
    return factors

def find_topo_order(proposal_factors):
    ## This function is aimed to find the sample order.
    G = nx.DiGraph()
    for index in proposal_factors:
        var = list(proposal_factors[index].var)
        if len(var) == 1:
            G.add_node(var[0])
        else:
            j = var.pop()

            for i in var:
                G.add_edge(i,j)
    nodes = nx.topological_sort(G)
    return nodes

def cal_joint_dist(proposal_factors):
    fac = Factor()
    for index in proposal_factors:
        fac = factor_product(fac,proposal_factors[index])
    return fac

def get_prob(samples, factor):
    sample_res = sorted(samples.items(), key=lambda d: d[0])
    sorted_samples_results = [value for key, value in sample_res]
    index = assignment_to_index(sorted_samples_results, factor.card)
    return factor.val[index]



""" END HELPER FUNCTIONS HERE """


def _sample_step(nodes, proposal_factors):
    """
    Performs one iteration of importance sampling where it should sample a sample for each node. The sampling should
    be done in topological order.

    Args:
        nodes: numpy array of nodes. nodes are sampled in the order specified in nodes
        proposal_factors: dictionary of proposal factors where proposal_factors[1] returns the
                sample distribution for node 1

    Returns:
        dictionary of node samples where samples[1] return the scalar sample for node 1.
    """
    samples = {}
    factors = proposal_factors.copy()
    """ YOUR CODE HERE: Use np.random.choice """
    for node in nodes:
        card = np.array(list(range(factors[node].card[0])))
        var = np.random.choice(a = card, p = factors[node].val)
        samples[node] = var
        factors = after_evidence(samples, factors)

    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples

def cal_prop_dist(nodes,proposal_factors):
    # I choose factors in the following way: factor.var = [0,1,2] means p(x2|x0, x1). So when we want to choose q(x2), \
    # we just need to find p(x2|x0, x1) = q(x2), we use factor.var[-1] == node
    fac = Factor()
    factors = proposal_factors.copy()
    for node in nodes:
        for index in factors:
            if node == factors[index].var[-1]:
                fac = factor_product(fac, factors[index])
                factors.pop(index)
                break
    return fac

def _get_conditional_probability(target_factors, proposal_factors, evidence, num_iterations):
    """
    Performs multiple iterations of importance sampling and returns the conditional distribution p(Xf | Xe) where
    Xe are the evidence nodes and Xf are the query nodes (unobserved).

    Args:
        target_factors: dictionary of node:Factor pair where Factor is the target distribution of the node.
                        Other nodes in the Factor are parent nodes of the node. The product of the target
                        distribution gives our joint target distribution.
        proposal_factors: dictionary of node:Factor pair where Factor is the proposal distribution to sample node
                        observations. Other nodes in the Factor are parent nodes of the node
        evidence: dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
        num_iterations: number of importance sampling iterations

    Returns:
        Approximate conditional distribution of p(Xf | Xe) where Xf is the set of query nodes (not observed) and
        Xe is the set of evidence nodes. Return result as a Factor
    """
    out = Factor()

    """ YOUR CODE HERE """
    # if evidence == {} I don't think we need to do the MC part, or you can invalid line 122-124.
    # if evidence == {}:
    #     out = cal_joint_dist(proposal_factors)
    #     return out

    nodes = find_topo_order(proposal_factors)

    #calculate proposal distribution
    evi_nodes = list(evidence.keys())
    for n in evi_nodes:
        nodes.remove(n)
    q_dist_factor = cal_prop_dist(nodes, proposal_factors)
    q_dist_factor = factor_evidence(q_dist_factor, evidence)
    # calculate the target distribution
    p_dist_factor = cal_joint_dist(target_factors)

    proposal_factors = after_evidence(evidence, proposal_factors)

    # r_i has a fixed number of value
    r_i = {}
    freq = {}
    all_assignments = q_dist_factor.get_all_assignments()
    for i in range(len(all_assignments)):
        index = assignment_to_index(all_assignments[i], q_dist_factor.card)
        q_prob = q_dist_factor.val[index]

        tmp_dict = {}
        for j in range(len(q_dist_factor.var)):
            tmp_dict[q_dist_factor.var[j]] = all_assignments[i][j]
        tmp_dict.update(evidence)
        p_prob = get_prob(tmp_dict, p_dist_factor)
        r_i[index] = p_prob/q_prob
        freq[index] = 0



    for _ in range(num_iterations):
        samples = _sample_step(nodes,proposal_factors)
        sample_res = sorted(samples.items(), key=lambda d: d[0])
        sorted_samples_results = [value for key, value in sample_res]
        index = assignment_to_index(sorted_samples_results, q_dist_factor.card)
        freq[index] += 1



    weights = [0]*len(all_assignments)
    sum_denominator = 0
    for i in range(len(weights)):
        sum_denominator += r_i[i]*freq[i]
    for i in range(len(weights)):
        weights[i] = r_i[i]/sum_denominator

    out_value = [0]*len(all_assignments)
    sum_denominator = 0
    for i in range(len(weights)):
        sum_denominator += weights[i]*freq[i]
    for i in range(len(weights)):
        out_value[i] = weights[i]*freq[i]/sum_denominator

    out.var = q_dist_factor.var
    out.card = q_dist_factor.card
    out.val = np.array(out_value)
    """ END YOUR CODE HERE """

    return out


def load_input_file(input_file: str) -> (Factor, dict, dict, int):
    """
    Returns the target factor, proposal factors for each node and evidence. DO NOT EDIT THIS FUNCTION

    Args:
        input_file: input file to open

    Returns:
        Factor of the target factor which is the target joint distribution of all nodes in the Bayesian network
        dictionary of node:Factor pair where Factor is the proposal distribution to sample node observations. Other
                    nodes in the Factor are parent nodes of the node
        dictionary of node:val pair where node is an evidence node while val is the evidence for the node.
    """
    with open(input_file, 'r') as f:
        input_config = json.load(f)
    target_factors_dict = input_config['target-factors']
    proposal_factors_dict = input_config['proposal-factors']
    assert isinstance(target_factors_dict, dict) and isinstance(proposal_factors_dict, dict)

    def parse_factor_dict(factor_dict):
        var = np.array(factor_dict['var'])
        card = np.array(factor_dict['card'])
        val = np.array(factor_dict['val'])
        return Factor(var=var, card=card, val=val)

    target_factors = {int(node): parse_factor_dict(factor_dict=target_factor) for
                      node, target_factor in target_factors_dict.items()}
    proposal_factors = {int(node): parse_factor_dict(factor_dict=proposal_factor_dict) for
                        node, proposal_factor_dict in proposal_factors_dict.items()}
    evidence = input_config['evidence']
    evidence = {int(node): ev for node, ev in evidence.items()}
    num_iterations = input_config['num-iterations']
    return target_factors, proposal_factors, evidence, num_iterations


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()
    # np.random.seed(0)

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    target_factors, proposal_factors, evidence, num_iterations = load_input_file(input_file=input_file)

    # solution part
    conditional_probability = _get_conditional_probability(target_factors=target_factors,
                                                           proposal_factors=proposal_factors,
                                                           evidence=evidence, num_iterations=num_iterations)
    print(conditional_probability)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    save__dict = {
        'var': np.array(conditional_probability.var).astype(int).tolist(),
        'card': np.array(conditional_probability.card).astype(int).tolist(),
        'val': np.array(conditional_probability.val).astype(float).tolist()
    }

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(save__dict, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
