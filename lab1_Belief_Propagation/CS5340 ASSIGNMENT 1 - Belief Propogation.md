# CS5340 ASSIGNMENT 1 - Belief Propogation

## Library requirement

matplotlib
networkx
numpy

## Assignment Description

This project calculate the factor tables to represent the conditional probability distributions of a Baysian Network. Detailed information is shown in ==lab1.pdf==.

Important functions include:

**1. Basic Factor Operations**

**factor_product():** This function should compute the product of two factors.
**factor_marginalize():** This function should sum over the indicated variable(s) and return the resulting factor.
**observe_evidence():** This function should take in a list of factors and the observed values of some of the variables, and modify the factors such that assignments not consistent with the observed values are set to zero.

**2. Naive Summation**

**compute_joint_distribution():** which computes the joint distribution over a Bayesian Network.

**3. Sum - Product Algorithm**

**compute_marginals_naive():** compute the marginal probabilities for a variable by marginalising out irrelevant variables from the joint distribution.

**compute_marignals_bp(): **computes the marginal probabilities of multiple variables using belief propagation.

**4. MAP inference using Max-product**

**factor_sum():** Analogous to factor_product(), but for log space.

**factor_max_marginalize(): **max over the indicated variable(s) and return the resulting factor.