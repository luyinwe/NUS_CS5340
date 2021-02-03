
from typing import Union

import numpy as np


class Factor:
    def __init__(self, var=None, card=None, val=None):

        if var is None:
            var = np.array([], np.int64)
        if card is None:
            card = np.array([], np.int64)
        if val is None:
            val = np.array([], np.float64)

        self.var = np.array(var, np.int64)
        self.card = np.array(card, np.int64)
        self.val = np.array(val, np.float64)

    def is_empty(self):
        """Returns true if the factor is empty (i.e. not initialized)"""
        return len(self.var) == 0

    def get_all_assignments(self):
        assignments = index_to_assignment(np.arange(int(np.prod(self.card))), self.card)
        return assignments

    def __repr__(self):
        if self.is_empty():
            str = 'Empty factor\n'
        else:
            str = 'Factor containing {} variables\n'.format(len(self.var))

            num_states = int(np.prod(self.card))
            assigments = index_to_assignment(np.arange(num_states), self.card)

            # Print header row
            header = '| ' + ' '.join(['X_{}'.format(i) for i in self.var]) + \
                     ' | Probability |'
            col_width = len(header)
            line = '-' * col_width + '\n'
            str += line + header + '\n' + line

            # Assignments
            for i in range(assigments.shape[0]):
                lhs = '   '.join(['{}'.format(a) for a in assigments[i]])
                row = '|  ' + lhs + '  | ' + '{:>11g}'.format(self.val[i]) + ' |\n'
                str = str + row
            str += line + '\n'
        return str

    def normalize(self):
        """Normalize the probablity such that it sums to 1.
        Use this function with care since not all factor tables should be
        normalized.
        """
        if not self.is_empty():
            self.val = self.val / np.sum(self.val)


def index_to_assignment(index: Union[int, np.ndarray], card: np.ndarray):
    """Convert index to variable assignment
    Args:
        index: Index to convert into assignment.
          If index is a vector of numbers, the function will return
          a matrix of assignments, one assignment per row.
        card: Cardinality of the factor
    """
    if isinstance(index, int):
        is_scalar = True
        index = np.array([index])
    else:
        is_scalar = False

    divisor = np.cumprod(np.concatenate([[1.], card[:-1]]))
    assignment = np.mod(
        np.floor(index[:, None] / divisor[None, :]),
        card[None, :]
    ).astype(np.int64)

    if is_scalar:
        assignment = assignment[0]  # Squeeze out row dimension

    return assignment


def assignment_to_index(assignment: np.ndarray, card: np.ndarray):
    """Convert assignment to index.
    Args:
        assignment: Assignment to convert to index
        card: Cardinality of the factor
    """
    multiplier = np.cumprod(np.concatenate([[1.], card[:-1]]))
    index = np.sum(assignment * multiplier, axis=-1).astype(np.int64)
    return index
