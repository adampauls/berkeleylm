This project provides a library for estimating storing large n-gram language models in memory and accessing them efficiently. It is described in <a href='http://nlp.cs.berkeley.edu/pubs/Pauls-Klein_2011_LM_paper.pdf'>this paper.</a> Its data structures are faster and smaller than <a href='http://nlp.cs.berkeley.edu/'>SRILM</a> and nearly as fast as <a href='http://kheafield.com/code/kenlm/'>KenLM</a> despite being written in Java instead of C++. It also achieves the best published lossless encoding of the Google n-gram corpus.

See <a href='http://berkeleylm.googlecode.com/svn/trunk/doc/edu/berkeley/nlp/lm/io/LmReaders.html'>here</a> for some documentation.

### News ###

December 6, 2014: Since Google has <a href='http://google-opensource.blogspot.com/2013/05/a-change-to-google-code-download-service.html'>deprecated downloads</a>, I will no longer be uploading new versions for the time being. You can build the same tarball that I create by running the "export" target on the <a href='https://code.google.com/p/berkeleylm/source/browse/trunk/build.xml'>build.xml</a> file in SVN.

June 9, 2013: version 1.1.5 has been released, which fixes a small bug with floating point rounding. Thanks to Giampiero Recco for finding this bug.

May 11, 2013: version 1.1.4 has been released, which fixes a bug with Kneser-Ney estimation on long sentences.

September 14, 2012: version 1.1.2 has been released, including some bug-fixes and documentation improvements. One particularly bad bug with stupid backoff LMs was fixed, so if you are using that code then please update. Also, binaries for Google n-gram-style LMs have been created from the <a href='http://books.google.com/ngrams/datasets'>Google Books</a> corpora. You can download them  [here](http://tomato.banatao.berkeley.edu:8080/berkeleylm_binaries).

July 26, 2012: version 1.1.0 has been released, including improved memory usage and thread-safe caching. Prior to version 1.1, to make caching threadsafe, the programmer had to ensure that each thread had its own local copy of a caching wrapper. Version 1.1 provides a thread-safe cache that internally manages thread-local caches using Java's `ThreadLocal` class. This incurs some performance overhead relative to the programmer manually ensuring thread-locality,  but it still significantly faster than not using the cache at all.

June 4, 2012: <a href='http://kheafield.com/professional/avenue/kenlm.pdf'>This paper</a> claims that Berkeley LM chops the mantissa of floats it stores to 12 bits. This is incorrect. This was inadvertent behavior that was fixed in 1.0b2. Because of the way BerkeleyLM encodes the floats it stores, correcting this behavior only added roughly an extra bit per value in our experiments. See [this erratum](Errata.md) for more details.

April 9, 2012: version 1.0.1 has been released. Fixes an occasional crash in estimation of Kneser-Ney models.

January 20, 2012: version 1.0.0 has been released. Fixes a bug in estimation of Kneser-Ney probabilities starting with the `<s>` tag. Also, several performance improvements, particularly in estimating Kneser-Ney probabilities. Note that binary compatibility was broken, so you will need to re-download all Google n-gram binaries.

August 14, 2011: version 1.0b3 has been released. This version can handle ARPA LM files which contain missing suffixes and prefixes. Also, we have released pre-built binaries for the Google N-Gram corpora. These can be downloaded [here](http://tomato.banatao.berkeley.edu:8080/berkeleylm_binaries).

June 24, 2011: version 1.0b2 has been released, with bug fixes from Kenneth Heafield, and some performance improvements.


