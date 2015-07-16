## Errata ##

The memory usages reported in [Pauls and Klein (2011)](http://nlp.cs.berkeley.edu/pubs/Pauls-Klein_2011_LM_paper.pdf) underestimate the true memory usages of the Kneser-Ney smoothed language models. This is due to inadvertently enabled rounding behavior, which rounded the mantissa of floats down from 23 bits to 12 bits. Because of the subtleties in how Berkeley LM stores floats, it is not the case that rounding saved 22 bits per n-gram  (11 for each of the probability and back-off). In fact, only one extra bit per n-gram was necessary. However, the model must store a list of all unique (probability, back-off) pairs, and the accidental rounding caused the number of unique pairs to be much smaller than necessary to store numbers up to true `float` precision.

Note that some subtle and boring things have changed in the Berkeley LM since the paper was written that further reduce memory usage, and the original version of the code would not be easy to reconstruct. In the following table, I provide the memory usages for the WMT2010 language model (1) as reported in the paper, using 12-bit rounding (2) as used by Berkeley LM 1.1.0, with 12-bit rounding, and (3) as used 1.1.0.

| **LM Type** | **PK2011 (12-bit rounding)** | **Version 1.1.0 (12-bit rounding)**| **Version 1.1.0 (no rounding)** |
|:------------|:-----------------------------|:-----------------------------------|:--------------------------------|
|Hash         | 7.5GB                        | 6.8GB                              | 7.5GB                           |
|Hash+Scroll  | 10.5GB                       | 9.9GB                              | 10.6GB                          |
|Sorted       | 5.5GB                        | (not implemented)                  | (not implemented)               |
|Compressed   | 3.7GB                        | 3.3GB                              | 4.3GB                           |