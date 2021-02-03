"""
Detailed description of the Factor data structure and related functions
-----------------------------------------------------------------------
Factors are implemented as a custom class. It consists of four main attributes
that you need to access/modify.

- 'var':  Variables
- 'card': Cardinality of the factor, one for each variable
- 'val': Probability
- 'val_argmax': To keep track of maximizing value (required only for MAP)

All the above attributes are numpy arrays, of type int64 for var and card, and
float64 for val.

You can instantiate an empty factor by calling the Factor() constructor without
any arguments. You can then fill up the above attributes. Alternatively,
you can also pass the relevant values during its construction, e.g.
>>>    phi = Factor(var=[1, 5],
                    card=[2, 3],
                    val=[0.6, 0.3, 0.2, 0.7, 0.2, 0.0])
If passed in as a list, the constructor will convert them into a numpy array.
The above code creates a factor phi over variables X_1 and X_5, var = [1, 5].
X_1 is binary valued, because phi.card[0] is 2. Similarly, X_5 has 3 states,
since phi.card[1] is 3.
A factor's values are stored in a row vector in .val field, using an ordering
such that the left-most variables as defined in the .var field cycle through
their values the fastest. More concretely, for the factor phi defined above,
we have the following mapping from variable assignments to the index of the
row vector in the .val field:

  ------------------------------
  | X_1 X_5 |    Probability   |
  ------------------------------
  |  0   0  | phi.val[0] = 0.6 |
  |  1   0  | phi.val[1] = 0.3 |
  |  0   1  | phi.val[2] = 0.2 |
  |  1   1  | phi.val[3] = 0.7 |
  |  0   2  | phi.val[4] = 0.2 |
  |  1   2  | phi.val[5] =   0 |
  ------------------------------

For easier debugging, you can visualize the factor values by calling
print(phi).

For your convenience, we have provided assignment_to_index() and
index_to_assignment() functions that compute the mapping between the assignments
A and the variable indices I, given D, the cardinality of the variables.

Concretely, given a factor phi, if phi.val(I) corresponds to the assignment A,
  I = assignment_to_index(A, D)
  A = index_to_assignment(I, D)
For instance, for the factor phi as defined above, row 4 contains the assignment
X_1=0, X_5=2. In this case, A = [0 2], I = 4, as phi.val[4] corresponds to the
value of phi(X_1=0, X_5=2).
Thus,
>>> assignment_to_index([0, 2], [2, 3])
will return 4, and
>>> index_to_assignment(4, [2, 3])
will return np.array([0, 2])

The above functions can also be called with 2D arrays for A, and a
1D array for I. In this case, the function will convert between multiple
assignments and indices. For example,
>>> index_to_assignment([0, 1, 2, 3, 4, 5], [2, 3])
will return the following 2D array:
  array([[0, 0],
         [1, 0],
         [0, 1],
         [1, 1],
         [0, 2],
         [1, 2]])

For convenience, you can also obtain the above array by calling
phi.get_all_assignments().


Acknowledgement:
  The factor class and this readme is inspired by the "Probabilistic Graphical
  Models" course assignments from Coursera.
    https://www.coursera.org/learn/probabilistic-graphical-models
"""

if __name__ == '__main__':
    from factor import *
    card = [2, 3]
    phi = Factor(var=[1, 5],
                 card=card,
                 val=[0.6, 0.3, 0.2, 0.7, 0.2, 0.0])

    # This will print the factor table, similar to the docstring above
    print(phi)

    # This should print 4, since phi.val[4] corresponds to the assignment
    # X_1=0, X_5=2.
    print(assignment_to_index(np.array([0, 2]), np.array(card)))
    print(assignment_to_index([0, 2], card))  # The functions accept a python list too

    # And this will print [0, 2]
    print(index_to_assignment(4, card))

    # This will print the entire 6x2 array of assignments
    print(index_to_assignment([0, 1, 2, 3, 4, 5], card))
