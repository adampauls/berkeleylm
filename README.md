# berkeleylm
Automatically exported from code.google.com/p/berkeleylm

This branched version 
1) converts the format of the project from Ant to Maven 
and 
2) rewrites nlp.lm.io/ArpaLmReader.java so that it's a bit shorter and reads all the files it used to, 
along with Sphinx (and I think, any other compliant ARPA) LM files.

The page at
http://www1.icsi.berkeley.edu/Speech/docs/HTKBook3.2/node213_mn.html
specifies the ARPA LM format. In particular, there doesn't seem to be a requirement for tabs, and
in fact the CMU Sphinx LM files at http://sourceforge.net/projects/cmusphinx/files/Acoustic%20and%20Language%20Models/
don't use tabs to delimit the logprob and backoff weight from the ngram.



the second more substantial change could probably be taken without taking the first build related one
