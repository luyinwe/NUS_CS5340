""" CS5340 Lab 2 Part 2: Parameter Learning
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

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')  # we will store the input data files here!
OBSERVATION_DIR = os.path.join(DATA_DIR, 'observations')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')


""" ADD HELPER FUNCTIONS HERE """

""" D ADD HELPER FUNCTIONS HERE """


def _learn_node_parameter_w(outputs, inputs=None):
    """
    Returns the weight parameters of the linear Gaussian [w0, w1, ..., wI], where I is the number of inputs. Students
    are encouraged to use numpy.linalg.solve() to get the weights. Learns weights for one node only.
    Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        inputs: N x I numpy array of input observations to the linear Gaussian model

    Returns:
        numpy array of (I + 1) weights [w0, w1, ..., wI]
    """
    num_inputs = 0 if inputs is None else inputs.shape[1]
    weights = np.zeros(shape=num_inputs + 1)

    """ YOUR CODE HERE """
    # Ax = b, A
    N_observe = outputs.shape[0]
    A = np.zeros(shape = (num_inputs+1, num_inputs+1))
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i==0 and j==0:
                A[i][j] = N_observe
            elif i==0 and j!=0:
                A[i][j] = np.sum(inputs[:,j-1])
            elif i!=0 and j==0:
                A[i][j] = np.sum(inputs[:,i-1])
            else:
                for k in range(N_observe):
                    A[i][j] += inputs[k,i-1]*inputs[k, j-1]
    b = np.zeros(shape=num_inputs + 1)
    for i in range(len(b)):
        if i==0:
            b[i] = np.sum(outputs)
        else:
            for k in range(N_observe):
                b[i] += inputs[k,i-1]*outputs[k]

    weights = np.linalg.solve(A, b)
    """ END YOUR CODE HERE """

    return weights


def _learn_node_parameter_var(outputs, weights, inputs):
    """
    Returns the variance i.e. sigma^2 for the node. Learns variance for one node only. Call once for each node.

    Args:
        outputs: numpy array of N output observations of the node
        weights: numpy array of (I + 1) weights of the linear Gaussian model
        inputs:  N x I numpy array of input observations to the linear Gaussian model.

    Returns:
        variance of the node's Linear Gaussian model
    """
    var = 0.

    """ YOUR CODE HERE """
    temp = 0
    N_observe = outputs.shape[0]
    if inputs is None:
        temp = np.sum((outputs-weights[0])**2)
    else:
        for i in range(N_observe):
            temp += (outputs[i] - (np.sum(weights[1:] * inputs[i]) +weights[0]))**2
    var = temp/N_observe



    """ END YOUR CODE HERE """

    return var


def _get_learned_parameters(nodes, edges, observations):
    """
    Learns the parameters for each node in nodes and returns the parameters as a dictionary. The nodes are given in
    ascending numerical order e.g. [1, 2, ..., V]

    Args:
        nodes: numpy array V nodes in the graph e.g. [1, 2, 3, ..., V]
        edges: numpy array of edges in the graph e.g. [i, j] implies i -> j where i is the parent of j
        observations: dictionary of node: observations pair where observations[1] returns a list of
                    observations for node 1.

    Returns:
        dictionary of parameters e.g.
        parameters = {
            "1": {  // first node
                "bias": w0 weight for node "1",
                "variance": variance for node "1"

                "2": weight for node "2", who is the parent of "1"
                ...
                // weights for other parents of "1"
            },
            ...
            // parameters of other nodes.
        }
    """
    parameters = {}

    """ YOUR CODE HERE """
    for node in nodes:
        parent_nodes = []
        for edge in edges:
            if node == edge[1]:
                parent_nodes.append(edge[0])
        output = np.array(observations[node])
        inputs = []
        for p_node in parent_nodes:
            inputs.append(observations[p_node])
        if inputs!=[]:
            inputs = np.array(inputs).T
        else:
            inputs = None
        weights = _learn_node_parameter_w(output, inputs)
        variance = _learn_node_parameter_var(output, weights, inputs)
        parameters[node] = {}
        parameters[node]["variance"] = variance
        parameters[node]["bias"] = weights[0]
        for p_node_index in range(len(parent_nodes)):
            parameters[node][parent_nodes[p_node_index]] = weights[p_node_index+1]
    """ END YOUR CODE HERE """

    return parameters


def main():
    """
    Helper function to load the observations, call your parameter learning function and save your results.
    DO NOT EDIT THIS FUNCTION.
    """
    argparser = ArgumentParser()
    argparser.add_argument('--case', type=int, required=True,
                           help='case number to create observations e.g. 1 if 1.json')
    args = argparser.parse_args()

    case = args.case
    observation_file = os.path.join(OBSERVATION_DIR, '{}.json'.format(case))
    with open(observation_file, 'r') as f:
         observation_config = json.load(f)

    nodes = observation_config['nodes']
    edges = observation_config['edges']
    observations = observation_config['observations']

    # solution part
    parameters = _get_learned_parameters(nodes=nodes, edges=edges, observations=observations)
    # end solution part

    # json only recognises floats, not np.float, so we need to cast the values into floats.
    for node, node_params in parameters.items():
        for param, val in node_params.items():
            node_params[param] = float(val)
        parameters[node] = node_params

    if not os.path.exists(PREDICTION_DIR):
        os.makedirs(PREDICTION_DIR)
    prediction_file = os.path.join(PREDICTION_DIR, '{}.json'.format(case))

    with open(prediction_file, 'w') as f:
        json.dump(parameters, f, indent=1)
    print('INFO: Results for test case {} are stored in {}'.format(case, prediction_file))


if __name__ == '__main__':
    main()
