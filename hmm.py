#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:58:21 2019

@author: liuyimo
"""
import numpy as np
# hidden states
states = ('low', 'high')
# observation states
observations = ('money', 'free', 'both')
# initial states and probability
start_probability = {'low': 0.6, 'high': 0.4}
# transiton probability
transition_probability = {
    'low': {'low': 0.7, 'high': 0.3},
    'high': {'low': 0.4, 'high': 0.6},
}
# observation probability
emission_probability = {
    'low': {'money': 0.5, 'free': 0.4, 'both': 0.1},
    'high': {'money': 0.3, 'free': 0.1, 'both': 0.6},
}
def simulate(T):

    def draw_from(probs):
        """
        1.np.random.multinomial:
        we use multinomial but no binomial because we may have more than two probabilities 
        >>> np.random.multinomial(20, [1/6.]*6, size=2)
                array([[3, 4, 3, 3, 4, 3],
                       [2, 4, 3, 4, 0, 7]])
         For the first run, we threw 3 times 1, 4 times 2, etc.  
         For the second, we threw 2 times 1, 4 times 2, etc.
        we only get 0 or 1 back. Therefore, we can use np.where to find the location 
        2.np.where:
        >>> x = np.arange(9.).reshape(3, 3)
        >>> np.where( x > 5 )
        (array([2, 2, 2]), array([0, 1, 2]))
        """
        return np.where(np.random.multinomial(1,probs) == 1)[0][0]

    observations = np.zeros(T, dtype=int)
    states = np.zeros(T, dtype=int)
    states[0] = draw_from(pi) # we put the initial probability at here
    observations[0] = draw_from(B[states[0],:])
# for each t, we generate the states values and observations values
    for t in range(1, T): 
        states[t] = draw_from(A[states[t-1],:])
        observations[t] = draw_from(B[states[t],:])
    return observations, states

 # convert observation from the dict to the array
def convert_map_to_vector(map_, label2id):
    
    v = np.zeros(len(map_), dtype=float)
    for e in map_:
        v[label2id[e]] = map_[e]
    return v

 # convert transition from the dict to the matrix
def convert_map_to_matrix(map_, label2id1, label2id2):
    
    m = np.zeros((len(label2id1), len(label2id2)), dtype=float)
    for line in map_:
        for col in map_[line]:
            m[label2id1[line]][label2id2[col]] = map_[line][col]
    return m
def generate_index_map(lables):
    id2label = {}
    label2id = {}
    i = 0
    for l in lables:
        id2label[i] = l
        label2id[l] = i
        i += 1
    return id2label, label2id
 
states_id2label, states_label2id = generate_index_map(states)
observations_id2label, observations_label2id = generate_index_map(observations)
print(states_id2label, states_label2id)
print(observations_id2label, observations_label2id)
# we got transition matrix
A = convert_map_to_matrix(transition_probability, states_label2id, states_label2id)
print(A)
# we got observation matrix
B = convert_map_to_matrix(emission_probability, states_label2id, observations_label2id)
print(B)
observations_index = [observations_label2id[o] for o in observations]
# we got the inistial probability
pi = convert_map_to_vector(start_probability, states_label2id)
print(pi)
# we simulate date 10 times and get different states/obserations values
observations_data, states_data = simulate(10)
print(observations_data)
print(states_data)
# we transfer the values to the label name
print("user's state: ", [states_id2label[index] for index in states_data])
print("user's observation: ", [observations_id2label[index] for index in observations_data])

# calculate the forward operator
def forward(obs_seq):
    N = A.shape[0]
    T = len(obs_seq)
    
    
    F = np.zeros((N,T))
    F[:,0] = pi * B[:, obs_seq[0]]
    # we can iterated calculate the forward operator by multiplying transtion prob and emmision prob on each t state 
    for t in range(1, T):
        for n in range(N):
            F[n,t] = np.dot(F[:,t-1], (A[:,n])) * B[n, obs_seq[t]]

    return F
# calculate the backward operator
def backward(obs_seq):
    N = A.shape[0]
    T = len(obs_seq)
    
    X = np.zeros((N,T))
    X[:,-1:] = 1

    for t in reversed(range(T-1)):
        for n in range(N):
            X[n,t] = np.sum(X[:,t+1] * A[n,:] * B[:, obs_seq[t+1]])

    return X
# We use EM method to calculate the parameter and likelihood
    """
    The EM means that we cannot directly calculate the full likelihood function but can 
    calculate the partially liklihood function conditioned by what we overserved until now.
    Therefore, we can execute the E step to find the likelihood at first ,then execute the M
    step to find the parameter. After we iterateive these two step unitl convergence,
    we can get the parameter and the full likelihood function. 
    """
def baum_welch_train(observations, A, B, pi, criterion=0.05):
    n_states = A.shape[0]
    n_samples = len(observations)

    done = False
    
        # E step
    while not done:
        """
        # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
        # Initialize alpha
        """
        alpha = forward(observations)
        """
        # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
        # Initialize beta
        """
        beta = backward(observations)
        """
        # ξ_t(i,j)=P(i_t=q_i,i_{i+1}=q_j|O,λ)
        xi means the prob from state q_i at time i to state q_j at time i+1, 
        condtioned by the gien observations
        """
        xi = np.zeros((n_states,n_states,n_samples-1))
        for t in range(n_samples-1):
            denom = np.dot(np.dot(alpha[:,t].T, A) * B[:,observations[t+1]].T, beta[:,t+1])
            for i in range(n_states):
                numer = alpha[i,t] * A[i,:] * B[:,observations[t+1]].T * beta[:,t+1].T
                xi[i,:,t] = numer / denom
        """"
        # γ_t(i)：gamma_t(i) = P(q_t = S_i | O, hmm)
        gamma means the prob that state q_i at time i
        """
        gamma = np.sum(xi,axis=1)
        # Need final gamma element for new B
        # because we only calcualte the sum of xi from 1 to n-1. we still need add one more to gamma
        prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
        gamma = np.hstack((gamma,  prod / np.sum(prod))) 
        
        # M step
        newpi = gamma[:,0]
        newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
        newB = np.copy(B)
        num_levels = B.shape[1]
        sumgamma = np.sum(gamma,axis=1)
        for lev in range(num_levels):
            mask = observations == lev
            newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma
        
        # check the criterion values
        if np.max(abs(pi - newpi)) < criterion and \
                        np.max(abs(A - newA)) < criterion and \
                        np.max(abs(B - newB)) < criterion:
            done = 1
        A[:], B[:], pi[:] = newA, newB, newpi
    return newA, newB, newpi

observations_data, states_data = simulate(100)
newA, newB, newpi = baum_welch_train(observations_data, A, B, pi)
print("newA: ", newA)
print("newB: ", newB)
print("newpi: ", newpi)
