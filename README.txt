Berkeley Language Model release 0.1

To compile this software just type "ant" in the root directory (assuming ant is installed and properly configured).

For examples of command-line usage of this software for manipulating language model files, see the examples/ directory.

Please see javadoc in edu.berkeley.nlp.lm.io.LmReaders file for documentation.

Known Issues:

When reading an ARPA LM file, this code assumes that for every n-gram, its (n-1)-gram suffix and prefix are both in the map. 
Some LM files violate this assumption, and so the code currently just ignores n-grams for which that condition is not satisfied.
In an upcoming release (probably mid-July), we will fix this limitation. 



