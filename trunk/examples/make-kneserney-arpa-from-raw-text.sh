#!/bin/bash

#estimate a 5-gram kneser-ney language model from the raw text file "big_test.txt"

java -ea -mx1000m -server -cp ../src edu.berkeley.nlp.lm.io.MakeKneserNeyArpaFromText 5 kneserNeyFromText.arpa ../test/edu/berkeley/nlp/lm/io/big_test.txt
