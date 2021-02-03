# taken from part 1
import copy
import numpy as np
from factor import Factor, index_to_assignment, assignment_to_index


def factor_product(A, B):
    """
    Computes the factor product of A and B e.g. A = f(x1, x2); B = f(x1, x3); out=f(x1, x2, x3) = f(x1, x2)f(x1, x3)
    Args:
        A: first Factor
        B: second Factor
    Returns:
        Returns the factor product of A and B
    """
    out = Factor()

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    if A.is_empty():
        return B
    if B.is_empty():
        return A

    out = Factor()

    # Set the variables of the output
    out.var = np.union1d(A.var, B.var)

    # Set the cardinality of the output
    out.card = np.zeros(len(out.var), np.int64)
    mapA = np.argmax(out.var[None, :] == A.var[:, None], axis=-1)
    mapB = np.argmax(out.var[None, :] == B.var[:, None], axis=-1)
    out.card[mapA] = A.card
    out.card[mapB] = B.card

    # Initialize the factor values to zero
    out.val = np.zeros(np.prod(out.card))
    assignments = out.get_all_assignments()
    idxA = assignment_to_index(assignments[:, mapA], A.card)
    idxB = assignment_to_index(assignments[:, mapB], B.card)

    # Populate the factor values
    out.val = A.val[idxA] * B.val[idxB]
    """ END YOUR CODE HERE """
    return out


def factor_marginalize(factor, var):
    """
    Returns factor after variables in var have been marginalized out.
    Args:
        factor: factor to be marginalized
        var: numpy array of variables to be marginalized over
    Returns:
        marginalized factor
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    out.var = np.setdiff1d(factor.var, var)
    map_out = np.argmax(out.var[:, None] == factor.var[None, :], axis=-1)
    out.card = factor.card[map_out]
    out.val = np.zeros(np.prod(out.card))

    assignments_in = index_to_assignment(np.arange(int(np.prod(factor.card))),
                                         factor.card)
    idx_out = assignment_to_index(assignments_in[:, map_out], out.card)

    for i in range(out.val.shape[0]):
        out.val[i] = np.sum(factor.val[idx_out == i])
    """ END YOUR CODE HERE """
    return out


def factor_evidence(factor, evidence):
    """
    Observes evidence and retains entries containing the observed evidence. Also removes the evidence random variables
    because they are already observed e.g. factor=f(1, 2) and evidence={1: 0} returns f(2) with entries from node1=0
    Args:
        factor: factor to reduce using evidence
        evidence:  dictionary of node:evidence pair where evidence[1] = evidence of node 1.
    Returns:
        Reduced factor that does not contain any variables in the evidence. Return an empty factor if all the
        factor's variables are observed.
    """
    out = copy.deepcopy(factor)

    """ YOUR CODE HERE,     HINT: copy from lab2 part 1! """
    # original code from lab1
    for (var, val) in evidence.items():
        if var in out.var:
            col_idx = np.argmax(out.var == var)
            assignments = out.get_all_assignments()
            mask = assignments[:, col_idx] == val
            out.val[~mask] = 0

    # difference: take only the relevant slices
    marg_var = [var for var in out.var if var in evidence.keys()]
    if len(marg_var) > 0:
        out = factor_marginalize(factor=out, var=np.array(marg_var))
    """ END YOUR CODE HERE """

    return out
