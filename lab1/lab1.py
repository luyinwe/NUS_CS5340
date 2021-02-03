""" CS5340 Lab 1: Belief Propagation and Maximal Probability
See accompanying PDF for instructions.

Name: <Lu, Yiwen>
Email: <e0576207>@u.nus.edu
Student ID: A0225573A
"""

import copy
from typing import List

import numpy as np

from factor import Factor, index_to_assignment, assignment_to_index, generate_graph_from_factors, \
    visualize_graph


"""For sum product message passing"""
def factor_product(A, B):
    """Compute product of two factors.

    Suppose A = phi(X_1, X_2), B = phi(X_2, X_3), the function should return
    phi(X_1, X_2, X_3)
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    out.val = A.val[idxA]*B.val[idxB]
    """ YOUR CODE HERE
    You should populate the .val field with the factor product
    Hint: The code for this function should be very short (~1 line). Try to
      understand what the above lines are doing, in order to implement
      subsequent parts.
    """
    return out


def factor_marginalize(factor, var):
    """Sums over a list of variables.

    Args:
        factor (Factor): Input factor
        var (List): Variables to marginalize out

    Returns:
        out: Factor with variables in 'var' marginalized out.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var
    """

    out.var = np.setxor1d(factor.var, np.array(var))

    for out_var in out.var:
        index = np.where(factor.var==out_var)
        out.card = np.append(out.card,factor.card[index])

    assignment = factor.get_all_assignments()
    index = []
    for single_var in var:
        index.append(np.where(factor.var==single_var))

    assignment = np.delete(assignment,index,axis=1)

    out.val = np.zeros(np.prod(out.card))
    for i in np.unique(assignment,axis=0):
        index_set = np.array(np.where(np.all(i==assignment,axis=1)))
        single_assignment = assignment_to_index(i,out.card)
        out.val[single_assignment] = np.sum(factor.val[index_set])
    return out


def observe_evidence(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    keys = []
    for i in evidence.keys():
        keys.append(i)

    for factor in out:
        if np.size(np.intersect1d(factor.var, np.array(keys))) ==0 :
            continue
        else:
            for key in keys:
                assignment = factor.get_all_assignments()
                if key not in factor.var:
                    continue
                evidence_var_index = np.where(factor.var==key)[0][0]
                map = np.where(assignment[:,evidence_var_index]!=evidence[key])[0]
                factor.val[map] = float(0)

    return out


"""For max sum meessage passing (for MAP)"""
def factor_sum(A, B):
    """Same as factor_product, but sums instead of multiplies
    """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    # Create output factor. Variables should be the union between of the
    # variables contained in the two input factors
    out = Factor()
    out.var = np.union1d(A.var, B.var)

    # Compute mapping between the variable ordering between the two factors
    # and the output to set the cardinality
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # For each assignment in the output, compute which row of the input factors
    # it comes from
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    """ YOUR CODE HERE
    You should populate the .val field with the factor sum. The code for this
    should be very similar to the factor_product().
    """
    out.val = A.val[idxA] + B.val[idxB]

    return out


def factor_max_marginalize(factor, var):
    """Marginalize over a list of variables by taking the max.

    Args:
        factor (Factor): Input factor
        var (List): Variable to marginalize out.

    Returns:
        out: Factor with variables in 'var' marginalized out. The factor's
          .val_argmax field should be a list of dictionary that keep track
          of the maximizing values of the marginalized variables.
          e.g. when out.val_argmax[i][j] = k, this means that
            when assignments of out is index_to_assignment[i],
            variable j has a maximizing value of k.
          See test_lab1.py::test_factor_max_marginalize() for an example.
    """
    out = Factor()

    """ YOUR CODE HERE
    Marginalize out the variables given in var. 
    You should make use of val_argmax to keep track of the location with the
    maximum probability.
    """
    out.var = np.setxor1d(factor.var, np.array(var))

    for out_var in out.var:
        index = np.where(factor.var == out_var)
        out.card = np.append(out.card, factor.card[index])

    assignment = factor.get_all_assignments()
    index = []
    out.val_argmax = []
    for single_var in var:
        index.append(np.where(factor.var == single_var))

    delete_assignment = np.delete(assignment, index, axis=1)

    out.val = np.zeros(np.prod(out.card))
    for i in np.unique(delete_assignment, axis=0):
        index_set = np.array(np.where(np.all(i == delete_assignment, axis=1)))
        index_set_max_index = np.argmax(factor.val[index_set])
        max_index_assignment = index_set[:,index_set_max_index][0]
        single_assignment = assignment_to_index(i, out.card)
        out.val[single_assignment] = factor.val[max_index_assignment]
        temp_dict = {}
        for single_var in var:
            index = np.argwhere(factor.var == single_var)[0][0]
            temp_dict[single_var] = assignment[max_index_assignment][index]
        out.val_argmax.append(temp_dict)
    return out


def compute_joint_distribution(factors):
    """Computes the joint distribution defined by a list of given factors

    Args:
        factors (List[Factor]): List of factors

    Returns:
        Factor containing the joint distribution of the input factor list
    """

    joint = factors[0]
    for factor_index in range(1, len(factors)):
        joint = factor_product(joint, factors[factor_index])
    """ YOUR CODE HERE
    Compute the joint distribution from the list of factors. You may assume
    that the input factors are valid so no input checking is required.
    """

    return joint


def compute_marginals_naive(V, factors, evidence):
    """Computes the marginal over a set of given variables

    Args:
        V (int): Single Variable to perform inference on
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k] = v indicates that
          variable k has the value v.

    Returns:
        Factor representing the marginals
    """

    output = Factor()

    # compute the joint distribution
    joint_distribution = compute_joint_distribution(factors)
    variables = factors[0].var
    for factor_index in range(1,len(factors)):
        variables = np.union1d(variables, factors[factor_index].var)

    evidence_keys = list(evidence.keys())
    evidence_keys.append(V)
    variables_need_marginalize = np.setxor1d(variables, np.array(evidence_keys))
    after_marginalize = factor_marginalize(joint_distribution, variables_need_marginalize)

    after_observe = observe_evidence([after_marginalize], evidence)[0]
    ##Normalize
    normalize_sum = np.sum(after_observe.val)
    after_observe.val /= normalize_sum

    # delete evidence, evidence value inequal to evidence needs to be delete, also card, also var
    assignment = after_observe.get_all_assignments()
    for evid in evidence:
        temp_value = evidence[evid]
        delete_index = np.array(np.where(after_observe.var==evid))[0][0]
        assignment_delete_index = np.array(np.where(assignment[:,delete_index] != temp_value))
        after_observe.var = np.delete(after_observe.var, delete_index)
        after_observe.card = np.delete(after_observe.card, delete_index)
        after_observe.val = np.delete(after_observe.val, assignment_delete_index, axis = 0)
        assignment = np.delete(assignment, assignment_delete_index, axis=0)
        assignment = np.delete(assignment, delete_index, axis=1)

    output = after_observe
    """ YOUR CODE HERE
    Compute the marginal. Output should be a factor.
    Remember to normalize the probabilities!
    """

    return output

def SendMessage(graph,j,i, msg):
    Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)

    if Neighbors_j == []:
        out = graph.edge[i][j]['factor']

    else:
        msg_product = msg[Neighbors_j[0]][j]
        for k in range(1,len(Neighbors_j)):
            msg_product = factor_product(msg_product, msg[Neighbors_j[k]][j])
        out = factor_product(graph.edge[i][j]['factor'], msg_product)
    if graph.node[j] != {}:
        out = factor_product(out, graph.node[j]['factor'])
    msg[j][i] = factor_marginalize(out,[j])

    return msg

def Collect(graph, i, j, msg):
    Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)

    for k in Neighbors_j:
        msg = Collect(graph,j,k, msg)
    msg = SendMessage(graph,j,i,msg)

    return msg

def Distribute(graph,i,j,msg):
    msg = SendMessage(graph, i, j, msg)
    Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)
    for k in Neighbors_j:
        msg = Distribute(graph, j, k, msg)
    return msg

def ComputeMarginal(graph,V, msg):
    output = []
    for i in V:
        Neighbors_i = graph.neighbors(i)
        msg_product = msg[Neighbors_i[0]][i]
        for j in range(1,len(Neighbors_i)):
            msg_product = factor_product(msg_product, msg[Neighbors_i[j]][i])
        if graph.node[i] != {}:
            msg_product = factor_product(msg_product, graph.node[i]['factor'])
        normalize_sum = np.sum(msg_product.val)
        msg_product.val /= normalize_sum
        output.append(msg_product)
    return output


def compute_marginals_bp(V, factors, evidence):
    """Compute single node marginals for multiple variables
    using sum-product belief propagation algorithm

    Args:
        V (List): Variables to infer single node marginals for
        factors (List[Factor]): List of factors representing the grpahical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        marginals: List of factors. The ordering of the factors should follow
          that of V, i.e. marginals[i] should be the factor for variable V[i].
    """
    # Dummy outputs, you should overwrite this with the correct factors
    marginals = []

    # Setting up messages which will be passed
    factors = observe_evidence(factors, evidence)
    graph = generate_graph_from_factors(factors)

    # Uncomment the following line to visualize the graph. Note that we create
    # an undirected graph regardless of the input graph since 1) this
    # facilitates graph traversal, and 2) the algorithm for undirected and
    # directed graphs is essentially the same for tree-like graphs.
    # visualize_graph(graph)

    # You can use any node as the root since the graph is a tree. For simplicity
    # we always use node 0 for this assignment.
    root = 0

    # Create structure to hold messages
    num_nodes = graph.number_of_nodes()
    messages = [[None] * num_nodes for _ in range(num_nodes)]

    # Collect
    Neighbors_root = graph.neighbors(root)
    for e in Neighbors_root:
        msg = Collect(graph,root,e, messages)
    for e in Neighbors_root:
        msg = Distribute(graph, root, e, msg)
    marginals = ComputeMarginal(graph,V,msg)
    """ YOUR CODE HERE
    Use the algorithm from lecture 4 and perform message passing over the entire
    graph. Recall the message passing protocol, that a node can only send a
    message to a neighboring node only when it has received messages from all
    its other neighbors.
    Since the provided graphical model is a tree, we can use a two-phase 
    approach. First we send messages inward from leaves towards the root.
    After this is done, we can send messages from the root node outward.
    
    Hint: You might find it useful to add auxilliary functions. You may add 
      them as either inner (nested) or external functions.
    """

    return marginals

def SendMessage_max(graph,j,i, prob, conf):
    Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)

    if Neighbors_j == []:
        out = graph.edge[i][j]['factor']

    else:
        prob_sum = prob[Neighbors_j[0]][j]
        for k in range(1,len(Neighbors_j)):
            prob_sum = factor_sum(prob_sum, prob[Neighbors_j[k]][j])
        out = factor_sum(graph.edge[i][j]['factor'], prob_sum)
    if graph.node[j] != {}:
        out = factor_sum(out, graph.node[j]['factor'])
    prob[j][i] = factor_max_marginalize(out, [j])

    conf[j][i] = {}
    assignments = out.get_all_assignments()
    i_index = np.where(out.var==i)[0][0]
    j_index = np.where(out.var == j)[0][0]
    for k in range(prob[j][i].card[0]):
        index = np.where(assignments[:,i_index]==k)[0]
        max_prob_index = np.argmax(out.val[index])
        change_to_exact_index = index[max_prob_index]
        j_exact_value = assignments[change_to_exact_index][j_index]
        conf[j][i][k] = j_exact_value

    return prob,conf

def Collect_max(graph, i, j, prob, conf):
    Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)

    for k in Neighbors_j:
        prob, conf = Collect_max(graph,j,k, prob, conf)
    prob, conf = SendMessage_max(graph,j,i,prob,conf)

    return prob, conf

def SetValue(i,j, conf, max_decoding):
    max_decoding[j] = conf[j][i][max_decoding[i]]
    # result = max_decoding[i]
    # i_index = np.where(conf[j][i].var == i)[0][0]
    # j_index = np.where(conf[j][i].var == j)[0][0]
    # assignments = conf[j][i].get_all_assignments()
    # compare_index = np.where(assignments[:,i_index]==result)[0]
    # max_index_i = np.argmax(conf[j][i].val[compare_index])
    # max_decoding[j] = assignments[max_index_i][j_index]
    # index = int(np.argmax(conf[j][i].val))
    # assignment = list(index_to_assignment(index, conf[j][i].card))
    # assignment.remove(conf[j][i].var[i_index])
    # max_decoding[j] = assignment[0]
    return max_decoding

def Distribute_max(graph,i,j, conf, max_decoding):
    max_decoding = SetValue(i,j,conf,max_decoding)

    Neighbors_j = graph.neighbors(j)
    Neighbors_j.remove(i)
    for k in Neighbors_j:
        max_decoding = Distribute_max(graph, j, k, conf, max_decoding)
    return max_decoding

def observe_evidence_max(factors, evidence=None):
    """Modify a set of factors given some evidence

    Args:
        factors (List[Factor]): List of input factors
        evidence (Dict): Dictionary, where the keys are the observed variables
          and the values are the observed values.

    Returns:
        List of factors after observing evidence
    """
    if evidence is None:
        return factors
    out = copy.deepcopy(factors)

    """ YOUR CODE HERE
    Set the probabilities of assignments which are inconsistent with the 
    evidence to zero.
    """
    keys = []
    for i in evidence.keys():
        keys.append(i)

    for factor in out:
        factor.val = np.log(factor.val)
        if np.size(np.intersect1d(factor.var, np.array(keys))) ==0 :
            continue
        else:
            for key in keys:
                if key not in factor.var:
                    continue
                assignment = factor.get_all_assignments()
                evidence_var_index = np.where(factor.var==key)[0][0]
                map = np.where(assignment[:,evidence_var_index]!=evidence[key])[0]
                factor.val[map] = -np.inf

    return out
def map_eliminate(factors, evidence):
    """Obtains the maximum a posteriori configuration for a tree graph
    given optional evidence

    Args:
        factors (List[Factor]): List of factors representing the graphical model
        evidence (Dict): Observed evidence. evidence[k]=v denotes that the
          variable k is assigned to value v.

    Returns:
        max_decoding (Dict): MAP configuration
        log_prob_max: Log probability of MAP configuration. Note that this is
          log p(MAP, e) instead of p(MAP|e), i.e. it is the unnormalized
          representation of the conditional probability.
    """

    max_decoding = {}
    log_prob_max = 0.0

    """ YOUR CODE HERE
    Use the algorithm from lecture 5 and perform message passing over the entire
    graph to obtain the MAP configuration. Again, recall the message passing 
    protocol.
    Your code should be similar to compute_marginals_bp().
    To avoid underflow, first transform the factors in the probabilities
    to **log scale** and perform all operations on log scale instead.
    You may ignore the warning for taking log of zero, that is the desired
    behavior.
    """
    factors = observe_evidence_max(factors, evidence)
    graph = generate_graph_from_factors(factors)

    num_nodes = graph.number_of_nodes()
    prob = [[None] * num_nodes for _ in range(num_nodes)]
    conf = [[None] * num_nodes for _ in range(num_nodes)]
    root = 0

    # Collect
    Neighbors_root = graph.neighbors(root)
    for e in Neighbors_root:
        prob,conf = Collect_max(graph, root, e, prob, conf)
    prob_sum = prob[Neighbors_root[0]][root]
    for j in range(1,len(Neighbors_root)):
        prob_sum = factor_sum(prob_sum,prob[Neighbors_root[j]][root])
    if graph.node[root]!={}:
        prob_sum = factor_sum(graph.node[root]['factor'],prob_sum)
    log_prob_max = np.max(prob_sum.val)
    max_decoding[root] = np.argmax(prob_sum.val)
    for e in Neighbors_root:
        max_decoding = Distribute_max(graph, root, e, conf, max_decoding)
    keys = []
    for i in evidence.keys():
        keys.append(i)
    for key in keys:
        del max_decoding[key]
    return max_decoding, log_prob_max