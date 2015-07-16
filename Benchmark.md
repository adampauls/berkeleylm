# Comparison with KenLM (still a draft) #

This is a brief comparison with [KenLM](http://kheafield.com/code/kenlm/), another recent and popular language modeling toolkit. If you don't want to see all the details, here is a quick summary:
  * For decoding tasks, where repeated queries are common, BerkeleyLM is slightly faster than KenLM, provided caching is enabled
  * For cache-unfriendly LM queries (like evaluating the perplexity of a corpus), KenLM is faster than BerkeleyLM
  * BerkeleyLM's `HASH` method takes about the same amount of memory as KenLM's `trie`, and BerkeleyLM's `HASH+SCROLL` uses a little bit more memory than KenLM's `probing`.
  * BerkeleyLM takes longer to load, but loading time small if you build a binary first for both KenLM and BerkeleyLM
  * BerkeleyLM offers more functionality than KenLM, including Kneser-Ney LM estimation from raw text custom handling of Google n-gram style count-based LMs.
  * Basically, we recommend using BerkeleyLM if you are programming in Java and KenLM if you are programming in C++.

## Timing ##

In this benchmark, I'll be using an evaluation similar to the one in [our paper](http://nlp.cs.berkeley.edu/pubs/Pauls-Klein_2011_LM_paper.pdf). Specifically, we logged all queries from the Joshua Decoder on one of their built-in text examples. We threw out all queries on rule fragments, because these queries are often short (unigrams and bigrams) and skew the results. Removing these queries leaves mostly 5-gram queries. We evaluate our models on the average time per query required to evaluate these queries in order. Note that this differs from [this benchmark](http://kheafield.com/code/kenlm/benchmark/), where performance is measured by computing the perplexity of a file. The latter is a cache-unfriendly task, but also one that end users probably don't care about, since you probably want to use these toolkits inside a decoder.

Here are some results. Lower times are faster. Each time is averaged over 3 runs. Variance isn't shown, but is usually on the order of +/- 10 ns/query. The BerkeleyLM models are shown with caching enabled. The model we used is the same 5-gram model using in [this Ken's benchmark](http://kheafield.com/code/kenlm/benchmark/).

| Model | Time (nanoseconds / query) |
|:------|:---------------------------|
| BerkeleyLM `HASH+SCROLL` | 82ns                       |
| BerkeleyLM `HASH` | 145ns                      |
| BerkeleyLM `COMPRESSED` | 670ns                      |
| KenLM `probing+state` | 109ns                      |
| KenLM `probing` | 187ns                      |
| KenLM `trie+state` | 200ns|

We have shown KenLM's `probing` model called in two different ways: the first (`+state`) involves using state that persists between queries, something which is also exploited by `HASH+SCROLL`. The second involves raw queries over n-grams, which is comparable to `HASH` since the latter does not need persistent state. Note that using BerkeleyLM or KenLM's persistent state makes integration into a decoder a little more involved, and in practice people tend to avoid using state so that the toolkit can be treated as black box. (Also note that although both toolkits use persistent state, these states actually quite different information).

Timings for `trie+state` are based on projections from Ken's benchmark (for some low-level reason, this would have been annoying to integrate into my test suite).

Basically, `HASH+SCROLL` is a little bit faster than `probing+state` and `HASH` is a little faster than `probing`. However, these differences in time are really quite tiny (a single, uncached memory access takes something like 30 ns on the machines I tested on), and you will probably not notice this difference when the model is actually decoding. Note that Ken's benchmark found that `probing` was not much faster than `trie` for decoding, despite a 2x difference in raw query speed.

We should point out that writing benchmarks that time things finally like this is very hard, since even a a single unnecessary memory access in the benchmark code can show up in the bench mark (it would usually add about 30ns). We have taken great care to optimize the benchmark to be favorable to all models, though some inefficiencies may still exist.

Finally, note that caching is a very important! Here, we show the results for BerkeleyLM with and without caching.

| Model | Caching | No Caching|
|:------|:--------|:----------|
| BerkeleyLM `HASH+SCROLL` | 82ns    | 244ns     |
| BerkeleyLM `HASH` | 145ns   | 515ns     |
| BerkeleyLM `COMPRESSED` | 670ns   | 4310ns    |

## Memory ##

Here are the memory usages for the models above, along with the query times.

| Model | Time (ns/query)  | Memory (GB) |
|:------|:-----------------|:------------|
| BerkeleyLM `HASH+SCROLL` | 82ns             | 4.0GB       |
| BerkeleyLM `HASH` | 145ns            | 2.9GB       |
| BerkeleyLM `COMPRESSED` | 510ns            | 1.5GB       |
| KenLM `probing+state` | 109ns            | 4.9GB       |
| KenLM `probing` | 187ns            | 4.9GB       |
| KenLM `trie+state` | 200ns            | 2.7GB       |

Note that the numbers above do not include storage of the words in the vocabulary. For BerkeleyLM, this storage takes an additional 0.2GB for each model. KenLM does not store the words in the vocabulary, keeping only vocabulary ids computing by hashing. Hence, KenLM's models can have hash collisions which would lead to incorrect query results, though in practice the odds of this are rare.

Also note that Ken's benchmark measures memory usage for BerkeleyLM by looking at memory used by the whole JVM, which is a little misleading, since the JVM does its own memory management and can sometimes keep around empty space owing to details of the implementation of garbage collection. The memory usages above count just the memory used by the data structures themselves.

## Loading Times ##

Here are the times for loading a language model from an ARPA file. Note that the variance is usually a few minutes.

| Model |Loading Time (min)  |
|:------|:-------------------|
| BerkeleyLM `HASH+SCROLL` | 24m                |
| BerkeleyLM `HASH` | 21 m               |
| BerkeleyLM `COMPRESSED` | 82m                |
| KenLM `probing+state` | 14m                |
| KenLM `probing` | 14m                |
| KenLM `trie+state` | 40m                |

KenLMs `trie` model and all BerkeleyLM models needs to make several passes over the data to pack things efficiently, which is why they are slower than `probing`. `COMPRESSED` must also compress the data, which adds significant overhead.

However, you should only incur these costs once. Loading from pre-built binaries is significantly faster. It takes a few seconds for BerkeleyLM to load this model, and KenLM is nearly instant because it uses memory mapping (though of course the cost of loading the binary from disk is still incurred, just spread out over execution).

## General Comparison ##

To sum up, here is a chart that compares the two models on functionality.

| Feature | BerkeleyLM | KenLM |
|:--------|:-----------|:------|
| LM Estimation | yes        | no    |
| Count-based language models | yes        | no    |
| Handles Google n-gram-formatted inputs | yes        | no    |
| Exact representation | yes        | almost|
| Quantization | can round floats to fixed number of bits | yes   |

We say that KenLM's models are "almost" exact because they do not (by default) keep the vocabulary around, so vocabulary hash collisions cannot be detected. KenLM's `probing` model additionally cannot detect collisions on n-gram hashes. In practice, collisions in either case are unlikely because hashes are stored to a large number of bits. However, it should be noted although `probing` (usually) answers queries correctly, the storage of n-gram hashes alone limits it in other ways. In particular, it is impossible to iterate over the n-grams in a `probing` model.