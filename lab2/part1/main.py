""" CS5340 Lab 2 Part 1: Junction Tree Algorithm
See accompanying PDF for instructions.

Name: <Lu, Yiwen>
Email: <e0576207>@u.nus.edu
Student ID: A0225573A
"""

import os
import numpy as np
import json
import networkx as nx
from argparse import ArgumentParser

from factor import Factor
from jt_construction import construct_junction_tree
from factor_utils import factor_product, factor_evidence, factor_marginalize

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
INPUT_DIR = os.path.join(DATA_DIR, 'inputs')  # we will store the input data files here!
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')  # we will store the prediction files here!


""" ADD HELPER FUNCTIONS HERE """
def Find_Neighbors(jt_edges, root):
    Neighbors_root = []
    for i in jt_edges:
        temp = i.copy()
        if root in temp:
            temp.remove(root)
            Neighbors_root.append(temp[0])
    return Neighbors_root

# def SendMessage(graph,j,i, msg):
def SendMessage(jt_cliques, jt_edges, jt_clique_factors, j, i, messages):
    Neighbors_j = Find_Neighbors(jt_edges,j)
    Neighbors_j.remove(i)

    if Neighbors_j==[]:
        out = jt_clique_factors[j]
    else:
        msg_product = Factor()
        for k in range(len(Neighbors_j)):
            msg_product = factor_product(msg_product,messages[Neighbors_j[k]][j])
        out = factor_product(jt_clique_factors[j],msg_product)

    Sij = list(set(jt_cliques[i]).intersection(jt_cliques[j]))
    margin_list = list(set(jt_cliques[j]).difference(set(Sij)))
    messages[j][i] = factor_marginalize(out,margin_list)
    return messages

def Collect(jt_cliques, jt_edges, jt_clique_factors, i, j, msg):

    Neighbors_j = Find_Neighbors(jt_edges,j)
    Neighbors_j.remove(i)

    for k in Neighbors_j:
        msg = Collect(jt_cliques, jt_edges, jt_clique_factors, j, k, msg)
    msg = SendMessage(jt_cliques, jt_edges, jt_clique_factors,j,i,msg)

    return msg

def Distribute(jt_cliques, jt_edges, jt_clique_factors,i,j,msg):
    msg = SendMessage(jt_cliques, jt_edges, jt_clique_factors, i, j, msg)
    # msg = SendMessage(graph, i, j, msg)
    Neighbors_j = Find_Neighbors(jt_edges, j)
    # Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)
    for k in Neighbors_j:
        msg = Distribute(jt_cliques, jt_edges, jt_clique_factors, j, k, msg)
    return msg

def ComputeMarginal(jt_cliques, jt_edges, jt_clique_factors, msg):
    output = []
    for i in range(len(jt_cliques)):
        Neighbors_i = Find_Neighbors(jt_edges,i)
        msg_product = Factor()
        for j in range(len(Neighbors_i)):
            msg_product = factor_product(msg_product, msg[Neighbors_i[j]][i])
        msg_product = factor_product(jt_clique_factors[i],msg_product)
        normalize_sum = np.sum(msg_product.val)
        msg_product.val = msg_product.val/(normalize_sum * 1.0)
        output.append(msg_product)
    return output

""" END HELPER FUNCTIONS HERE """


def _update_mrf_w_evidence(all_nodes, evidence, edges, factors):
    """
    Update the MRF graph structure from observing the evidence

    Args:
        all_nodes: numpy array of nodes in the MRF
        evidence: dictionary of node:observation pairs where evidence[x1] returns the observed value of x1
        edges: numpy array of edges in the MRF
        factors: list of Factors in teh MRF

    Returns:
        numpy array of query nodes
        numpy array of updated edges (after observing evidence)
        list of Factors (after observing evidence; empty factors should be removed)
    """

    query_nodes = all_nodes
    updated_edges = edges
    updated_factors = factors

    """ YOUR CODE HERE """
    updated_factors = []
    for factor in factors:
        updated_factors.append(factor_evidence(factor,evidence))

    evidence_list = list(evidence.keys())
    delete_list = []
    for evi in evidence_list:
        query_nodes = np.setdiff1d(query_nodes,np.array(evi))
        for edge_index in range(len(edges)):
            if evi in list(edges[edge_index]):
                delete_list.append(edge_index)
    updated_edges = np.delete(updated_edges,delete_list,axis=0)
    """ END YOUR CODE HERE """

    return query_nodes, updated_edges, updated_factors


def _get_clique_potentials(jt_cliques, jt_edges, jt_clique_factors):
    """
    Returns the list of clique potentials after performing the sum-product algorithm on the junction tree

    Args:
        jt_cliques: list of junction tree nodes e.g. [[x1, x2], ...]
        jt_edges: numpy array of junction tree edges e.g. [i,j] implies that jt_cliques[i] and jt_cliques[j] are
                neighbors
        jt_clique_factors: list of clique factors where jt_clique_factors[i] is the factor for cliques[i]

    Returns:
        list of clique potentials computed from the sum-product algorithm
    """
    clique_potentials = jt_clique_factors

    """ YOUR CODE HERE """
    root = 0

    # Create structure to hold messages
    num_nodes = len(jt_cliques)
    msg = [[None] * num_nodes for _ in range(num_nodes)]

    # Collect
    Neighbors_root = Find_Neighbors(jt_edges, root)

    for e in Neighbors_root:
        msg = Collect(jt_cliques, jt_edges, jt_clique_factors, root, e, msg)
    for e in Neighbors_root:
        msg = Distribute(jt_cliques, jt_edges, jt_clique_factors, root, e, msg)
    clique_potentials = ComputeMarginal(jt_cliques, jt_edges, jt_clique_factors, msg)
    """ END YOUR CODE HERE """

    assert len(clique_potentials) == len(jt_cliques)
    return clique_potentials


def _get_node_marginal_probabilities(query_nodes, cliques, clique_potentials):
    """
    Returns the marginal probability for each query node from the clique potentials.

    Args:
        query_nodes: numpy array of query nodes e.g. [x1, x2, ..., xN]
        cliques: list of cliques e.g. [[x1, x2], ... [x2, x3, .., xN]]
        clique_potentials: list of clique potentials (Factor class)

    Returns:
        list of node marginal probabilities (Factor class)

    """
    query_marginal_probabilities = []

    """ YOUR CODE HERE """
    # first compute the joint contribution first
    for i in range(len(query_nodes)):
        min_len = 0
        for j in range(len(cliques)):
            if query_nodes[i] in cliques[j]:
                if min_len>len(cliques[j]) or min_len==0:
                    min_len = len(cliques[j])
                    min_index = j
        margin_list = cliques[min_index].copy()
        margin_list.remove(query_nodes[i])
        query_marginal_probabilities.append(factor_marginalize(clique_potentials[min_index],margin_list))


    """ END YOUR CODE HERE """

    return query_marginal_probabilities


def get_conditional_probabilities(all_nodes, evidence, edges, factors):
    """
    Returns query nodes and query Factors representing the conditional probability of each query node
    given the evidence e.g. p(xf|Xe) where xf is a single query node and Xe is the set of evidence nodes.

    Args:
        all_nodes: numpy array of all nodes (random variables) in the graph
        evidence: dictionary of node:evidence pairs e.g. evidence[x1] returns the observed value for x1
        edges: numpy array of all edges in the graph e.g. [[x1, x2],...] implies that x1 is a neighbor of x2
        factors: list of factors in the MRF.

    Returns:
        numpy array of query nodes
        list of Factor
    """
    query_nodes, updated_edges, updated_node_factors = _update_mrf_w_evidence(all_nodes=all_nodes, evidence=evidence,
                                                                              edges=edges, factors=factors)

    jt_cliques, jt_edges, jt_factors = construct_junction_tree(nodes=query_nodes, edges=updated_edges,
                                                               factors=updated_node_factors)

    clique_potentials = _get_clique_potentials(jt_cliques=jt_cliques, jt_edges=jt_edges, jt_clique_factors=jt_factors)

    query_node_marginals = _get_node_marginal_probabilities(query_nodes=query_nodes, cliques=jt_cliques,
                                                            clique_potentials=clique_potentials)

    return query_nodes, query_node_marginals


def parse_input_file(input_file: str):
    """ Reads the input file and parses it. DO NOT EDIT THIS FUNCTION. """
    with open(input_file, 'r') as f:
        input_config = json.load(f)

    nodes = np.array(input_config['nodes'])
    edges = np.array(input_config['edges'])

    # parse evidence
    raw_evidence = input_config['evidence']
    evidence = {}
    for k, v in raw_evidence.items():
        evidence[int(k)] = v

    # parse factors
    raw_factors = input_config['factors']
    factors = []
    for raw_factor in raw_factors:
        factor = Factor(var=np.array(raw_factor['var']), card=np.array(raw_factor['card']),
                        val=np.array(raw_factor['val']))
        factors.append(factor)
    return nodes, edges, evidence, factors


def main():
    """ Entry function to handle loading inputs and saving outputs. DO NOT EDIT THIS FUNCTION. """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    input_file = os.path.join(INPUT_DIR, '{}.json'.format(case))
    nodes, edges, evidence, factors = parse_input_file(input_file=input_file)

    # solution part:
    query_nodes, query_conditional_probabilities = get_conditional_probabilities(all_nodes=nodes, edges=edges,
                                                                                 factors=factors, evidence=evidence)

    predictions = {}
    for i, node in enumerate(query_nodes):
        probability = query_conditional_probabilities[i].val
        predictions[int(node)] = list(np.array(probability, dtype=float))

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))
    with open(prediction_file, 'w') as f:
        json.dump(predictions, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
