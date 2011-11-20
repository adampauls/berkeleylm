#!/bin/bash


# build a compressed stupid-backoff language model binary from the Google n-grams dir "googledir"

# a "googledir" should be in the format 
# 1gms/vocab_cs.gz [here, vocab_cs.gz should have the unigram frequencies sorted in decreasing order of frequency]
# 2gms/2gm-0001.gz 2gm-0002.gz É
# 3gms/3gm-0001.gz É 
# see ../test/edu/berkeley/nlp/lm/io/googledir for an example
java -ea -mx1000m -server -cp ../src edu.berkeley.nlp.lm.io.MakeLmBinaryFromGoogle ../test/edu/berkeley/nlp/lm/io/googledir google.binary
