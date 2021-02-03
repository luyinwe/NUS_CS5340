""" CS5340 Lab 3: Hidden Markov Models
See accompanying PDF for instructions.

Name: <Lu, Yiwen>
Email: <e0576207>@u.nus.edu
Student ID: A0225573A
"""
import numpy as np
import scipy.stats
from scipy.special import softmax
from sklearn.cluster import KMeans


def initialize(n_states, x):
    """Initializes starting value for initial state distribution pi
    and state transition matrix A.

    A and pi are initialized with random starting values which satisfies the
    summation and non-negativity constraints.
    """
    seed = 5340
    np.random.seed(seed)

    pi = np.random.random(n_states)
    A = np.random.random([n_states, n_states])

    # We use softmax to satisify the summation constraints. Since the random
    # values are small and similar in magnitude, the resulting values are close
    # to a uniform distribution with small jitter
    pi = softmax(pi)
    A = softmax(A, axis=-1)

    # Gaussian Observation model parameters
    # We use k-means clustering to initalize the parameters.
    x_cat = np.concatenate(x, axis=0)
    kmeans = KMeans(n_clusters=n_states, random_state=seed).fit(x_cat[:, None])
    mu = kmeans.cluster_centers_[:, 0]
    std = np.array([np.std(x_cat[kmeans.labels_ == l]) for l in range(n_states)])
    phi = {'mu': mu, 'sigma': std}

    return pi, A, phi

def calculate_pro_x_z(n_states, input ,phi):
    pro_x_z = np.zeros([len(input),n_states])
    for i in range(n_states):
        pro_x_z[:,i] = scipy.stats.norm.pdf(input,loc = phi['mu'][i], scale = phi['sigma'][i])

    return pro_x_z

def forward_pass(x_list, pi, A, phi):
    n_states = pi.shape[0]
    alpha_list = [np.zeros([len(x), n_states]) for x in x_list]
    C_n = [np.zeros(len(x)) for x in x_list]
    pro_x_z = [np.zeros([len(x), n_states]) for x in x_list]

    for alpha_index in range(len(alpha_list)):
        # calculate the initial value
        pro_x_z[alpha_index] = calculate_pro_x_z(n_states, x_list[alpha_index], phi)
        tmp = pi * pro_x_z[alpha_index][0]
        C_n[alpha_index][0] = np.sum(tmp)
        alpha_list[alpha_index][0, :] = tmp/np.sum(tmp)

        # alpha(z_n) = p(x_n|z_n)sum(alpha(z_(n-1)*prob(z_n|z_(n-1)))
        for i in range(1,len(alpha_list[alpha_index])):
            tmp = np.matmul(alpha_list[alpha_index][i-1,:], A)
            tmp = tmp * pro_x_z[alpha_index][i]
            C_n[alpha_index][i] = np.sum(tmp)
            alpha_list[alpha_index][i,:] = tmp/C_n[alpha_index][i]
    return alpha_list, C_n, pro_x_z

def backward_pass(x_list, pi, A, phi, C_n, pro_x_z):
    n_states = pi.shape[0]
    beta_list = [np.zeros([len(x), n_states]) for x in x_list]

    for beta_index in range(len(beta_list)):
        beta_list[beta_index][-1, :] = np.ones(n_states)
        N = len(beta_list[beta_index]) - 2
        while(N>=0):
            tmp = beta_list[beta_index][N+1,:]* pro_x_z[beta_index][N+1]
            beta_list[beta_index][N,:] = np.sum(np.multiply(tmp,A), axis = 1)/C_n[beta_index][N+1]
            N-=1
    return beta_list


"""E-step"""
def e_step(x_list, pi, A, phi):
    """ E-step: Compute posterior distribution of the latent variables,
    p(Z|X, theta_old). Specifically, we compute
      1) gamma(z_n): Marginal posterior distribution, and
      2) xi(z_n-1, z_n): Joint posterior distribution of two successive
         latent states

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        pi (np.ndarray): Current estimated Initial state distribution (K,)
        A (np.ndarray): Current estimated Transition matrix (K, K)
        phi (Dict[np.ndarray]): Current estimated gaussian parameters

    Returns:
        gamma_list (List[np.ndarray]), xi_list (List[np.ndarray])
    """
    n_states = pi.shape[0]
    gamma_list = [np.zeros([len(x), n_states]) for x in x_list]
    xi_list = [np.zeros([len(x)-1, n_states, n_states]) for x in x_list]

    """ YOUR CODE HERE
    Use the forward-backward procedure on each input sequence to populate 
    "gamma_list" and "xi_list" with the correct values.
    Be sure to use the scaling factor for numerical stability.
    """
    alpha_list, C_n, pro_x_z = forward_pass(x_list, pi, A, phi)
    beta_list = backward_pass(x_list, pi, A, phi, C_n, pro_x_z)

    for gamma_index in range(len(gamma_list)):
        gamma_list[gamma_index] = alpha_list[gamma_index]*beta_list[gamma_index]

    for xi_index in range(len(xi_list)):
        for n in range(1,len(xi_list[xi_index])+1):
            tmp = C_n[xi_index][n]**(-1)* alpha_list[xi_index][n-1]
            tmp = tmp[:,np.newaxis]
            tmp = np.multiply(tmp,A)
            tmp = np.multiply(tmp,pro_x_z[xi_index][n])
            tmp = np.multiply(tmp,beta_list[xi_index][n])
            xi_list[xi_index][n-1] = tmp

    return gamma_list, xi_list


"""M-step"""
def m_step(x_list, gamma_list, xi_list):
    """M-step of Baum-Welch: Maximises the log complete-data likelihood for
    Gaussian HMMs.
    
    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        gamma_list (List[np.ndarray]): Marginal posterior distribution
        xi_list (List[np.ndarray]): Joint posterior distribution of two
          successive latent states

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.
    """

    n_states = gamma_list[0].shape[1]
    pi = np.zeros([n_states])
    A = np.zeros([n_states, n_states])
    phi = {'mu': np.zeros(n_states),
           'sigma': np.zeros(n_states)}

    """ YOUR CODE HERE
    Compute the complete-data maximum likelihood estimates for pi, A, phi.
    """
    gamma_arr = np.array(gamma_list)
    pi_denorminator = np.sum(gamma_arr[:, 0])
    for i in range(len(pi)):
        pi_numerator = np.sum(gamma_arr[:,0,i])
        pi[i] = pi_numerator/pi_denorminator

    xi_arr = np.array(xi_list)
    for i in range(len(A)):
        A_denorminator = np.sum(xi_arr[:, :, i, :])
        for j in range(len(A[0])):
            A_numerator = np.sum(xi_arr[:,:,i,j])
            A[i][j] = A_numerator/A_denorminator

    x_arr = np.array(x_list)
    for i in range(n_states):
        phi['mu'][i] = np.sum(gamma_arr[:,:,i]*x_arr)/np.sum(gamma_arr[:,:,i])
        phi['sigma'][i] = np.sqrt(
            np.sum(gamma_arr[:, :, i] * (x_arr - phi['mu'][i])**2) / np.sum(
                gamma_arr[:, :, i]))

    return pi, A, phi


"""Putting them together"""
def fit_hmm(x_list, n_states):
    """Fit HMM parameters to observed data using Baum-Welch algorithm

    Args:
        x_list (List[np.ndarray]): List of sequences of observed measurements
        n_states (int): Number of latent states to use for the estimation.

    Returns:
        pi (np.ndarray): Initial state distribution
        A (np.ndarray): Time-independent stochastic transition matrix
        phi (Dict[np.ndarray]): Parameters for the Gaussian HMM model, contains
          two fields 'mu', 'sigma' for the mean and standard deviation
          respectively.

    """

    # We randomly initialize pi and A, and use k-means to initialize phi
    # Please do NOT change the initialization function since that will affect
    # grading
    pi, A, phi = initialize(n_states, x_list)

    """ YOUR CODE HERE
     Populate the values of pi, A, phi with the correct values. 
    """
    while(1):
        phi_old = phi
        gamma_list, xi_list = e_step(x_list,pi,A,phi)
        pi, A, phi = m_step(x_list, gamma_list, xi_list)
        if (np.abs(phi_old['mu'] - phi['mu'])< 10**(-4)).all() and (np.abs(phi_old['sigma'] - phi['sigma'])< 10**(-4)).all():
            break

    return pi, A, phi

