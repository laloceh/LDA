#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:54:48 2019

@author: eduardo

https://github.com/cuteboydot/Latent-Dirichlet-Allocation
"""

import numpy as np
import random
import sys

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=50)

'''
    Process the Gibbs Sampling
    Input: data without the token information, the token information, the token id    
'''
def gibbs_proc(word_doc_topic_task, sample, idx):

    # make Topic-Doc relation 
    # Theta
    for a in range(docs.shape[0]):
        for b in range(topics):
            count = np.count_nonzero((word_doc_topic_task[:, 1] == str(a)) & (word_doc_topic_task[:, 2] == str(b)))
            theta_num[a][b] = count + alpha

    for a in range(docs.shape[0]):
        for b in range(topics):
            count = np.sum(theta_num[a])
            theta_prob[a][b] = float(theta_num[a][b])/float(count)

    # make word-topic relation
    # Phi
    for a in range(words_uniq.shape[0]):
        for b in range(topics):
            count = np.count_nonzero((word_doc_topic_task[:, 0] == str(words_uniq[a])) & (word_doc_topic_task[:, 2] == str(b)))
            phi_num[a][b] = count + beta

    for a in range(words_uniq.shape[0]):
        for b in range(topics):
            count = np.sum(phi_num[a])
            phi_prob[a][b] = float(phi_num[a][b])/float(count)

    del word_doc_topic_task

    # allocate topic-word
    # Z
    # sample [word, doc num, topic num, word uniq idx]
    if idx >= 0 :
        p_post = np.zeros((topics))
        for a in range(topics):
            #print 'a:',a
            #print 'sample:',sample
            #print 'sample[1]',sample[1]
            #print theta_prob
            p_topic_doc = theta_prob[int(sample[1])][a]
            #print theta_prob[int(sample[1])][a]

            topic_tot = np.sum((phi_num.T)[a])
            #print "phi_num[a]",np.sum(phi_num[:,a])
            #print "topic tot",topic_tot
            p_word_topic = phi_num[int(sample[3])][a]/topic_tot
            p_post[a] = p_topic_doc * p_word_topic

        topic_max = np.argmax(p_post)
        return topic_max



alpha = 0.1
beta = 0.001
topics = 2
epoch = 200

docs = np.array(("basketball is a team sport",
                "the five basketball players fall into five positions",
                "basketball was originally played with a soccer ball",
                "auto racing has existed since the invention of the automobile.",
                "the first racing of two self-powered road vehicles",
                "the largest stock car racing governing body is nascar (national association for stock car auto racing)."))


if __name__ == "__main__":

    words_full = []
    words_uniq = []
    doc_word = np.zeros((docs.shape[0]))
    doc_words_size = np.zeros((docs.shape[0]))
    a = 0

    for doc in docs:
        doc_words = doc.split()
        words_full += doc_words
        doc_words_size[a] = len(doc_words)
        a += 1
    
    
    words_full = np.array(words_full)
    print ("words_full: All the words")
    print (words_full)
    
    print ("doc_words_size: Number of words per document")
    print (doc_words_size)
       
    words = np.array(list(set(words_full)))
    words_uniq = np.unique(words_full)
    words_uniq = np.reshape(words_uniq, (words_uniq.shape[0]))
    print ("words_uniq")
    print (words_uniq)
    
    # word, doc num, topic num, unique word index
    word_doc_topic = np.array(['keyword', 0, 0, 0])
    a=0
    for doc in docs:
        words = doc.split()
        for word in words:
            id_uniq = np.where(words_uniq == word)[0]
            to = random.randrange(0, topics)
            element = (word, a, to, id_uniq[0])
            #print "Element", element
            word_doc_topic = np.vstack((word_doc_topic, element))
        a += 1
            
    # Remove the first entry, it is the "keyword" row.
    word_doc_topic = word_doc_topic[1:, :]
    
    print ("\nword_doc_topic")
    print "Word - Doc - Topic - Word_ID"
    print (word_doc_topic)
    
    print
    # Topic distribution for documents
    # Docs x Topics
    print "Docs %d, Topics %d" % (docs.shape[0], topics)
    theta_num = np.zeros((docs.shape[0], topics))
    theta_prob = np.zeros((docs.shape[0], topics))
    #print theta_num.shape
    #print theta_prob.shape
    # Word distribution for topics
    # Words x Topics
    print "Words %d, Topics %d" % (words_uniq.shape[0], topics)
    phi_num = np.zeros((words_uniq.shape[0], topics))
    phi_prob = np.zeros((words_uniq.shape[0], topics))
    #print phi_num.shape
    #print phi_prob.shape
    
    
    # do gibbs sampling proc
    for a in range(epoch):
        for b in range(word_doc_topic.shape[0]):

            word_doc_topic_task = word_doc_topic.copy()
            
            # get the whole row for the token
            sample = word_doc_topic_task[b]

            # Remove the token
            word_doc_topic_task = np.delete(word_doc_topic_task, b, axis=0)

            # word_doc_topic_task is the same minus the token
            # send data without token, the sample of token, the token ID
            topic_max = gibbs_proc(word_doc_topic_task, sample, b)
            
            word_doc_topic[b][2] = topic_max
            del word_doc_topic_task
        
    # print final state
    gibbs_proc(word_doc_topic, [None, None, None, None], -1)
    
    print ("~~~~RESULTS~~~~")
    print ("theta P(Topic;Doc)")
    for a in range(theta_num.shape[0]) :
        print ("Doc%d => %s = %s" % (a, str(theta_num[a]), str(theta_prob[a])))

    print
    print ("phi P(Word;Topic)")
    for a in range(phi_num.shape[0]) :
        print ("%s => %s = %s" % (words_uniq[a], str(phi_num[a]), str(phi_prob[a])))

    print
    print ("word_doc_topic")
    print (word_doc_topic)