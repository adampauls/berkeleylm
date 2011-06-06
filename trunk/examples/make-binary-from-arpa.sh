#!/bin/bash


#build a hash-based language model binary from the file big_test.arpa
java -ea -mx1000m -server -cp ../src edu.berkeley.nlp.lm.io.MakeLmBinaryFromArpa ../test/edu/berkeley/nlp/lm/io/big_test.arpa big_test.binary
