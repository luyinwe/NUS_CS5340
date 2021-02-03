import json
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Factor:
    def __init__(self, var=None, card=None, val=None, val_argmax=None):

        if var is None:
            var = np.array([], np.int64)
        if card is None:
            card = np.array([], np.int64)
        if val is None:
            val = np.array([], np.float64)

        self.var = np.array(var)
        self.card = np.array(card)
        self.val = np.array(val)

        # You should use this field for passing messages about the maximizing
        # values for MAP inference. This should contain a list of the same
        # length as val, where each element is a dictionary of the form {k: v},
        # where v is the maximizing value of the variable k.
        self.val_argmax = val_argmax

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

    def __eq__(self, other):
        """Checks whether two factors are the same.
        Note: Does not check the argmax field
        """
        if set(self.var) != set(other.var):
            return False

        # Finds the mapping for variable ordering
        map_other = np.argmax(self.var[None, :] == other.var[:, None], axis=-1)

        if not np.all(self.card[map_other] == other.card):
            return False

        self_assignments = self.get_all_assignments()
        other_assignments = self_assignments[:, map_other]
        other_index = assignment_to_index(other_assignments, other.card)
        if np.allclose(self.val, other.val[other_index]):
            return self.val_argmax == other.val_argmax
        else:
            return False


def index_to_assignment(index, card):
    """Convert index to variable assignment. See factor_readme.py for details.

    Args:
        index: Index to convert into assignment.
          If index is a vector of numbers, the function will return
          a matrix of assignments, one assignment per row.
        card: Cardinality of the factor
    """
    if isinstance(index, int):
        is_scalar = True
        index = [index]
    else:
        is_scalar = False

    # Handle case where a list is passed in instead of numpy array
    index = np.array(index)
    card = np.array(card)

    divisor = np.cumprod(np.concatenate([[1.], card[:-1]]))
    assignment = np.mod(
        np.floor(index[:, None] / divisor[None, :]),
        card[None, :]
    ).astype(np.int64)

    if is_scalar:
        assignment = assignment[0]  # Squeeze out row dimension

    return assignment


def assignment_to_index(assignment, card):
    """Convert assignment to index. See factor_readme.py for details.

    Args:
        assignment: Assignment to convert to index
        card: Cardinality of the factor
    """

    # Handle case where a list is passed in instead of numpy array
    assignment = np.array(assignment)
    card = np.array(card)

    multiplier = np.cumprod(np.concatenate([[1.], card[:-1]]))
    index = np.sum(assignment * multiplier, axis=-1).astype(np.int64)
    return index


def generate_graph_from_factors(factors):
    """Generates a graph from the factors. Only supports pairwise factors
    Args:
        factors (List[Factor]): List of factors. For this assignment, all
          factors will be either unary or pairwise.

    Returns:
        nx.Graph instance.
    """
    G = nx.Graph()
    for factor in factors:
        if len(factor.var) == 1:
            G.add_node(factor.var[0], factor=factor)
        elif len(factor.var) == 2:
            G.add_edge(factor.var[0], factor.var[1], factor=factor)
        else:
            raise NotImplementedError('Not supported')
    return G


def visualize_graph(graph):
    nx.draw_networkx(graph, with_labels=True, font_weight='bold',
                     node_size=1000, arrowsize=20)
    plt.axis('off')
    plt.show()


def to_factor(data):
    factor = Factor(var=data['var'],
                    card=data['card'],
                    val=data['val'])
    return factor


def load_factor_list_from_json(fname):
    """Parses the factor list from a json file"""
    with open(fname, 'r') as f:
        input_config = json.load(f)
        factor_data = input_config['factors']
        factor_list = [to_factor(f) for f in factor_data]
    return factor_list


def write_factor_list_to_json(factor_list, fname):

    with open(fname, 'w') as fid:
        fid.write('{\n')
        fid.write('    "factors": [\n')

        for i in range(len(factor_list)):
            factor = factor_list[i]
            var_str = ', '.join(['{}'.format(v) for v in factor.var.tolist()])
            card_str = ', '.join(['{}'.format(c) for c in factor.card.tolist()])
            val_str= ', '.join(['{:.8f}'.format(v) for v in factor.val.tolist()])
            fid.write('        {\n')
            fid.write('            "var": [' + var_str + '],\n')
            fid.write('            "card": [' + card_str + '],\n')
            fid.write('            "val": [' + val_str + ']\n')
            fid.write('        },\n' if i < len(factor_list) - 1 else '        }\n')

        fid.write('    ]\n')
        fid.write('}\n')
