#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:34:35 2019

@author: eduardo
"""

############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Text Mining

# Citation: 
# PEREIRA, V. (2017). Project: LDA - Latent Dirichlet Allocation, File: Python-DM-Text Mining-01.py, GitHub repository: 
# <https://github.com/Valdecy/Latent-Dirichlet-Allocation>

############################################################################

# Installing Required Libraries
import numpy  as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from random import randint

import sys

# Function: lda_tm
def lda_tm(document = [], K = 2, alpha = 0.12, eta = 0.01, iterations = 5000, dtm_matrix = False, dtm_bin_matrix = False, dtm_tf_matrix = False, dtm_tfidf_matrix = False, co_occurrence_matrix = False, correl_matrix = False):
   
    ################ Part 1 - Start of Function #############################
    tokenizer = RegexpTokenizer(r'\w+')
    result_list = []    # This will hold all the results
   
    # Corpus
    corpus = []
    for i in document:
        tokens = tokenizer.tokenize(i.lower())
        corpus.append(tokens)
    #print corpus
        
    # Corpus ID
    corpus_id = []
    for i in document:
        tokens = tokenizer.tokenize(i.lower())
        corpus_id.append(tokens)
    #print corpus_id
        
    # Unique Words
    uniqueWords = []
    for j in range(0, len(corpus)): 
        for i in corpus[j]:
            if not i in uniqueWords:
                uniqueWords.append(i)
    
    #print uniqueWords
    
    # Corpus ID for Unique Words   
    # Convert documents to sequence of numbers
    for j in range(0, len(corpus)): 
        for i in range(0, len(uniqueWords)):
            for k in range(0, len(corpus[j])): 
                if uniqueWords[i] == corpus[j][k]:
                    corpus_id[j][k]  = i  
    
    #print corpus_id

    # Topic Assignment
    # this will hold the topic assignment for each word in the documents
    topic_assignment = []
    for i in document:
        tokens = tokenizer.tokenize(i.lower())
        topic_assignment.append(tokens)
    
    #print topic_assignment

    # dtm (Document x Words matrix )
    if dtm_matrix == True or dtm_bin_matrix == True or dtm_tf_matrix == True or dtm_tfidf_matrix == True or co_occurrence_matrix == True or correl_matrix == True:
        dtm = np.zeros(shape = (len(corpus), len(uniqueWords)))   
        for j in range(0, len(corpus)): 
            for i in range(0, len(uniqueWords)):
                for k in range(0, len(corpus[j])): # For each word in each document
                    if uniqueWords[i] == corpus[j][k]:  # Append 1 to the word in the document
                        dtm[j][i]  = dtm[j][i] + 1
        dtm_pd = pd.DataFrame(dtm, columns = uniqueWords)   # Convert the array to a Pandas DataFrame
    
    if dtm_matrix == True:
        result_list.append(dtm_pd)
    
    #print dtm_pd
        
    # dtm_bin
    # This is a one-hot encoded representation for each document
    if dtm_bin_matrix == True or co_occurrence_matrix == True or correl_matrix == True:
        dtm_bin = np.zeros(shape = (len(corpus), len(uniqueWords)))  
        for i in range(0, len(corpus)): 
            for j in range(0, len(uniqueWords)):
                if dtm[i,j] > 0:
                    dtm_bin[i,j] = 1
        dtm_bin_pd = pd.DataFrame(dtm_bin, columns = uniqueWords)
    
    if dtm_bin_matrix == True:
        result_list.append(dtm_bin_pd)
    
    #print dtm_bin_pd
    #print
    #sys.exit(0)
        
    # dtm_tf
    # This is term frequency for each word.
    if dtm_tf_matrix == True:
        dtm_tf = np.zeros(shape = (len(corpus), len(uniqueWords))) 
        for i in range(0, len(corpus)): 
            for j in range(0, len(uniqueWords)):
                if dtm[i,j] > 0:
                    dtm_tf[i,j] = dtm[i,j]/dtm[i,].sum()
        dtm_tf_pd = pd.DataFrame(dtm_tf, columns = uniqueWords)
        result_list.append(dtm_tf_pd)
    
    #print dtm_tf_pd
        
    # dtm_tfidf
    if dtm_tfidf_matrix == True:
        idf  = np.zeros(shape = (1, len(uniqueWords)))  
        for i in range(0, len(uniqueWords)):
            idf[0,i] = np.log10(dtm.shape[0]/(dtm[:,i]>0).sum())
        dtm_tfidf = np.zeros(shape = (len(corpus), len(uniqueWords)))
        for i in range(0, len(corpus)): 
            for j in range(0, len(uniqueWords)):
                dtm_tfidf[i,j] = dtm_tf[i,j]*idf[0,j]
        dtm_tfidf_pd = pd.DataFrame(dtm_tfidf, columns = uniqueWords)
        result_list.append(dtm_tfidf_pd)
    
    #print dtm_tfidf_pd
        
    # Co-occurrence Matrix
    if co_occurrence_matrix == True:
        co_occurrence = np.dot(dtm_bin.T,dtm_bin)
        co_occurrence_pd = pd.DataFrame(co_occurrence, columns = uniqueWords, index = uniqueWords)
        result_list.append(co_occurrence_pd)
    
    #print co_occurrence_pd
    
    # Correlation Matrix
    if correl_matrix == True:
        correl = np.zeros(shape = (len(uniqueWords), len(uniqueWords)))
        for i in range(0, correl.shape[0]): 
            for j in range(i, correl.shape[1]):
                correl[i,j] = np.corrcoef(dtm_bin[:,i], dtm_bin[:,j])[0,1]
        correl_pd = pd.DataFrame(correl, columns = uniqueWords, index = uniqueWords)
        result_list.append(correl_pd) 
   
    #print correl_pd   

    
    # LDA Initialization
    
    # Ramdomly assign a topic to each work
    #print topic_assignment[0][0] # This is the word "data" in the first document
    #print topic_assignment
    for i in range(0, len(topic_assignment)): # for each document
        for j in range(0, len(topic_assignment[i])): # for each word in that document
            topic_assignment[i][j]  = randint(0, K-1) # randomly pick a topic number
    
    #print topic_assignment
    print
    print "Initial topic assignment for each word in each document"
    for d, document in enumerate(topic_assignment):
        print d+1, document
    #sys.exit(0)
    
    # This is the Document-Topic matrix, Theta()        
    #print len(topic_assignment)
    cdt = np.zeros(shape = (len(topic_assignment), K))
    for i in range(0, len(topic_assignment)): 
        for j in range(0, len(topic_assignment[i])): 
            for m in range(0, K): 
                if topic_assignment[i][j] == m:
                    cdt[i][m]  = cdt[i][m] + 1  # Add 1 for every word assigned to one topic
    

    print
    print "Document Topic count distribution"
    for i in range( len( corpus ) ) :
        print i+1, cdt[i]

    
    # This is the word-topic matrix , Phi()
    cwt = np.zeros(shape = (K,  len(uniqueWords)))
    for i in range(0, len(corpus)):     # For each document
        for j in range(0, len(uniqueWords)):    # For each unique word
            for m in range(0, len(corpus[i])):  # for each word in each document
                if uniqueWords[j] == corpus[i][m]: # if the word is in the vocabulary
                    for n in range(0, K):   # Check which is its assigned topic
                        if topic_assignment[i][m] == n:
                            cwt[n][j]  = cwt[n][j] + 1  # Increase the assignment
    
    print
    print "There are %d words in the vocabulary" % len(uniqueWords)
    print "Topic - Word matrix"
    print cwt
    
    # LDA Algorithm
    print
    print "Sampling for %d iterations ...." % iterations
    for i in range(0, iterations + 1):  # For each iteration
        for d in range(0, len(corpus)): # For each document
            for w in range(0, len(corpus[d])): # For each word
                initial_t = topic_assignment[d][w]  # Get word's current topic assignment
                
                if (i == 0) and (d==0) and (w==0):
                    print "Example: First word of first document initially has assigned topic=", initial_t
                    
                word_num = corpus_id[d][w]

                # Decrement this word's topic count in the matrices
                cdt[d,initial_t] = cdt[d,initial_t] - 1 
                cwt[initial_t,word_num] = cwt[initial_t,word_num] - 1

                # Calculate the probability distribution for each topic for this word
                p_z = ((cwt[:,word_num] + eta) / (np.sum((cwt), axis = 1) + len(corpus) * eta)) * \
                        ((cdt[d,] + alpha) / (sum(cdt[d,]) + K * alpha ))
                        
                if (i==0) and (d==0) and (w==0):
                    print "The probability distribution for this word is ", p_z
                        
                z = np.sum(p_z)
                p_z_ac = np.add.accumulate(p_z/z) 
                if (i==0) and (d==0) and (w==0):
                    print "The accumulated probability distribution for this word is ", p_z_ac
                    
                u = np.random.random_sample()
                if (i==0) and (d==0) and (w==0):
                    print "Random sample =", u
                
                # Calculate the new topic assignment
                for m in range(0, K):
                    if u <= p_z_ac[m]:
                        final_t = m
                        break
                
                topic_assignment[d][w] = final_t 
                if (i==0) and (d==0) and (w==0):
                    print "The new topic assigned to this word is ", final_t
                
                # Add this new topic assignment to the matrices to be used
                # in the next iteration
                cdt[d,final_t] = cdt[d,final_t] + 1 
                cwt[final_t,word_num] = cwt[final_t,word_num] + 1


        if i % 100 == 0:
            print('Iteration:', i)
    
    print "Sampling finished"
    
    
    print 
    print "Calculate the two final distributions"
    print "Theta"
    theta = (cdt + alpha)
    for i in range(0, len(theta)): 
        for j in range(0, K):
            theta[i,j] = theta[i,j]/np.sum(theta, axis = 1)[i]

    print theta
    result_list.append(theta)
    
    print 
    print "Phi"        
    phi = (cwt + eta)
    d_phi = np.sum(phi, axis = 1)
    for i in range(0, K): 
        for j in range(0, len(phi.T)):
            phi[i,j] = phi[i,j]/d_phi[i]
     
    phi_pd = pd.DataFrame(phi.T, index = uniqueWords)
    print phi_pd
    
    result_list.append(phi_pd)
    
    return result_list

    ############### End of Function ##############

######################## Part 2 - Usage ####################################

# Documents
doc_1 = "data mining technique data mining first favourite technique"
doc_2 = "data mining technique data mining second favourite technique"
doc_3 = "data mining technique data mining third favourite technique"
doc_4 = "data mining technique data mining fourth favourite technique"
doc_5 = "friday play guitar"
doc_6 = "saturday will play guitar"
doc_7 = "sunday will play guitar"
doc_8 = "monday will play guitar"
doc_9 = "good good indeed can thank"

# Compile Documents
docs = [doc_1, doc_2, doc_3, doc_4, doc_5, doc_6, doc_7, doc_8, doc_9]

# Call Function
lda = lda_tm(document = docs, K = 3, alpha = 0.12, eta = 0.01, iterations = 2500, co_occurrence_matrix = True,
                 dtm_matrix = True, dtm_bin_matrix = True, dtm_tf_matrix = True, dtm_tfidf_matrix = True, 
                 correl_matrix = True)

########################## End of Code #####################################
