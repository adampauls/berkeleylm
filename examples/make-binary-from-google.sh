#!/bin/bash


#build a compressed stupid-backoff language model binary from the Google n-grams dir "googledir"
java -ea -mx1000m -server -cp ../src edu.berkeley.nlp.lm.io.MakeLmBinaryFromGoogle ../test/edu/berkeley/nlp/lm/io/googledir google.binary
