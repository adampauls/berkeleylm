#!/bin/bash


#first, build a hash-based language model binary from the file big_test.arpa
java -ea -mx1000m -server -cp ../src edu.berkeley.nlp.lm.io.MakeLmBinaryFromArpa ../test/edu/berkeley/nlp/lm/io/big_test.arpa big_test.binary

#now score a sample sentence
echo "This is a sample sentence ." | java -ea -mx1000m -server -cp ../src edu.berkeley.nlp.lm.io.ComputeLogProbabilityOfTextStream  big_test.binary
