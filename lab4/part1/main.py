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
from factor_utils import factor_evidence, factor_product, assignment_to_index, index_to_assignment
from factor import Factor
from argparse import ArgumentParser
from tqdm import tqdm

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """



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

    """ YOUR CODE HERE: Use np.random.choice """

    evidence={}
    fac = Factor()
    for node in nodes: 
        fac = factor_evidence(proposal_factors[node], evidence)
        n = np.random.choice(fac.card[0], 1, p = fac.val)
        samples[node] = n
        evidence[node] = n
            

    """ END YOUR CODE HERE """

    assert len(samples.keys()) == len(nodes)
    return samples


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

    nodes = target_factors.keys() #get all nodes in p

    ne_proposal_factors = {} #eliminate the evidence
    for node, pro_factor in proposal_factors.items():
        if node in evidence.keys():
            continue
        pro_factor = factor_evidence(pro_factor, evidence)
        ne_proposal_factors[node] = pro_factor

    ne_nodes = ne_proposal_factors.keys() #get the nodes does not contain variables in evidence


    #get basic config of out
    out = target_factors[0]
    #print("?",out)
    i = 0
    for node, factor in target_factors.items():
        if i == 0: 
            i = i + 1
            continue
        out = factor_product(out,factor)

    out = factor_evidence(out,evidence)
    #print("?",out)
    config_num = len(out.val)
    r_list = np.zeros(config_num)
    r = np.zeros(num_iterations)
    #w = np.zeros(num_iterations)
    count = np.zeros(config_num)

    for i in range(config_num):
        ass = index_to_assignment(i, out.card)
        j = 0
        dic = {}
        for node in ne_nodes:
            #print(ne_nodes)
            dic[node] = ass[j]
            j = j + 1
        #print(ass)
        q = 1.00000000000
        for node in ne_nodes:
            ass1 = [] 
            #print("!",node)
            for var in ne_proposal_factors[node].var:
                #print("?",var)
                ass1.append(dic[var])
            
            idx = assignment_to_index(ass1, ne_proposal_factors[node].card)
            q = q * ne_proposal_factors[node].val[idx]

        p = 1.00000000000
        for node in nodes:
            ass2 = [] 
            for var in target_factors[node].var:
                if var in evidence.keys():
                    ass2.append(evidence[var])
                else :
                    ass2.append(dic[var])

             
            idx = assignment_to_index(ass2, target_factors[node].card)
            p = p * target_factors[node].val[idx]
            #print("p_",p)

        r_list[i] = p/q

    r_sum = 0.00000000
    for num_idx in range(num_iterations):

        samples = _sample_step(ne_nodes, ne_proposal_factors)
        #print(ne_proposal_factors)
        ass3 = np.zeros(len(samples.keys()))
        i = 0
        for node in ne_nodes:
            ass3[i] = samples[node]
            i = i + 1

        idx = assignment_to_index(ass3, out.card)
        count[idx] = count[idx] + 1
        r[num_idx] = r_list[idx]
        
        r_sum = r_sum + r[num_idx]

    #compute p(x_F|x_E)
    out.val = count * r_list / r_sum



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
