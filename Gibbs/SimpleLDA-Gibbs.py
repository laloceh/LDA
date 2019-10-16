#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:11:40 2019

@author: eduardo

https://wiseodd.github.io/techblog/2017/09/07/lda-gibbs/
"""
    
import numpy as np
from numpy.random import randint, dirichlet
import sys

# Words
W = np.array([0, 1, 2, 3, 4])

# D := document words
X = np.array([
    [0, 0, 1, 2, 2],
    [0, 0, 1, 1, 1],
    [0, 1, 2, 2, 2],
    [4, 4, 4, 4, 4],
    [3, 3, 4, 4, 4],
    [3, 4, 4, 4, 4]
])

N_D = X.shape[0]  # num of docs
N_W = W.shape[0]  # num of words
N_K = 2  # num of topics

#N_D = 5
#N_W = 10
#N_K = 2

alpha = 0.1
gamma = 0.02

iterations  = 1000

def initialize_Z(N_D, N_W):
    Z = np.zeros(shape=[N_D, N_W])
    
    for i in range(N_D):
        for l in range(N_W):
            Z[i,l] = np.random.randint(N_K)
    return Z    
    

def initialize_Pi(N_D, N_K, alpha):
    Pi = np.zeros([N_D, N_K])
    for i in range(N_D):
        Pi[i] = np.random.dirichlet(alpha * np.ones(N_K))
        
    return Pi

def initialize_B(N_K, N_W, gamma):
    B = np.zeros([N_K, N_W])
    for k in range(N_K):
        B[k] = np.random.dirichlet(gamma * np.ones(N_W))
    
    return B

        
def doGibbs(Z, Pi, B, alpha, gamma, iterations, N_D, N_W, N_K, X):
    print "Do Gibbs %d iterations " % iterations
    
    for it in range(iterations):
            
        # Sample from Z
        #----------------------
        for i in range(N_D):
            for v in range(N_W):
                #Calculate parameters for Z
                p_iv = np.exp( np.log( Pi[i]) + np.log( B[:, X[i, v]]) )
                p_iv /= np.sum(p_iv)
                
                # Resample word topic assignment Z
                Z[i,v] = np.random.multinomial(1, p_iv).argmax()
        
        
        # Sample from full conditional of Pi
        # ----------------------------
        for i in range(N_D):
            m = np.zeros(N_K)
            
            # Gather statistics
            for k in range(N_K):
                m[k] = np.sum(Z[i] == k)    # how many words in the documents belong to each class
                
            # Resample doc topic ditribution
            Pi[i, :] = np.random.dirichlet( alpha + m) # Add alpha so it is not 0 (some probab)
            
        
        # Sample from full conditional of B
        #--------------------------------
        for k in range(N_K):
            n = np.zeros(N_W)
            
            # Gather statistics
            for v in range(N_W):
                for i in range(N_D):
                    for l in range(N_W):
                        n[v] += (X[i, l] == v) and (Z[i, l] == k)
            
           
            B[k,:] = np.random.dirichlet( gamma + n)
            
            
#############################
if __name__ == "__main__":
    print "Total documents %d, total words %d, total topics %d" % (N_D, N_W, N_K)
    
    
    Z = initialize_Z(N_D, N_W)
    print "Document-Word[Topic] assignment"
    print "Z"
    print Z   
        
    
    # Pi: document topic distribution
    Pi = initialize_Pi(N_D, N_K, alpha)
    print "Document-Topic distribution"
    print "Pi"
    print Pi
    
    # B = word topic distribution
    B = initialize_B(N_K, N_W, gamma)
    print "Topic-Word distribution"
    print "B"
    print B
    
    doGibbs(Z, Pi, B, alpha, gamma, iterations, N_D, N_W, N_K, X)
    
    #print Z
    #print Pi
    #print B
    
    print "Document-Topic distribution:"
    print Pi
    
    print
    print "Topic-Word distribution"
    print B
    
