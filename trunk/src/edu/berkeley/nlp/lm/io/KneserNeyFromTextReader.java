package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap.Entry;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.StrUtils;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

public class KneserNeyFromTextReader<W>
{

	private static final float D = 0.6f;

	public static <W> HashNgramMap<KneserNeyCounts> readFromFiles(List<File> files, WordIndexer<W> wordIndexer, int maxOrder) {
		final Iterable<String> allLinesIterator = Iterators.flatten(new Iterators.Transform<File, Iterator<String>>(files.iterator())
		{

			@Override
			protected Iterator<String> transform(File file) {
				try {
					return IOUtils.lineIterator(file.getPath());
				} catch (IOException e) {
					throw new RuntimeException(e);

				}
			}
		});

		return countNgrams(wordIndexer, maxOrder, allLinesIterator);

	}

	/**
	 * @param <W>
	 * @param wordIndexer
	 * @param maxOrder
	 * @param allLinesIterator
	 * @param ngrams
	 * @return
	 */
	static <W> HashNgramMap<KneserNeyCounts> countNgrams(WordIndexer<W> wordIndexer, int maxOrder, final Iterable<String> allLinesIterator) {
		final KneseryNeyCountValueContainer values = new KneseryNeyCountValueContainer(maxOrder);
		HashNgramMap<KneserNeyCounts> ngrams = HashNgramMap.createExplicitWordHashNgramMap(values, new ConfigOptions(), maxOrder, false);
		values.initForMap(ngrams);
		for (String line : allLinesIterator) {
			final String[] words = line.split(" ");
			int[] sent = new int[words.length + 2];
			sent[0] = wordIndexer.getOrAddIndex(wordIndexer.getStartSymbol());
			sent[sent.length - 1] = wordIndexer.getOrAddIndex(wordIndexer.getEndSymbol());
			for (int i = 0; i < words.length; ++i) {
				sent[i + 1] = wordIndexer.getOrAddIndexFromString(words[i]);
			}
			for (int ngramOrder = 0; ngramOrder < maxOrder; ++ngramOrder) {
				for (int i = 0; i < sent.length; ++i) {
					if (i - ngramOrder < 0) continue;
					ngrams.put(sent, i - ngramOrder, i + 1, null);
				}
			}
		}
		return ngrams;
	}

	public static <W> void writeToFile(String file, WordIndexer<W> wordIndexer, HashNgramMap<KneserNeyCounts> ngrams) {
		PrintWriter out = IOUtils.openOutHard(file);
		writeToPrintWriter(wordIndexer, ngrams, out);

	}

	/**
	 * @param <W>
	 * @param wordIndexer
	 * @param ngrams
	 * @param out
	 */
	static <W> void writeToPrintWriter(WordIndexer<W> wordIndexer, HashNgramMap<KneserNeyCounts> ngrams, PrintWriter out) {
		int lmOrder = ngrams.getMaxLmOrder();
		out.println();
		out.println("\\data\\");
		writeHeader(ngrams, lmOrder, out);
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			out.println("\\" + (ngramOrder + 1) + "-grams:");
			for (Entry<KneserNeyCounts> entry : ngrams.getNgramsForOrder(ngramOrder)) {
				ProbBackoffPair val = ngramOrder == lmOrder - 1 ? getHighestOrderProb(entry.key, entry.value, ngrams) : getLowerOrderProb(entry.key, 0,
					entry.key.length, ngrams, lmOrder);
				final String ngramString = StrUtils.join(WordIndexer.StaticMethods.toList(wordIndexer, entry.key));
				float prob = val.prob + getLowerOrderProb(entry.key, 0, entry.key.length - 1, ngrams, lmOrder).backoff
					* recurse(entry.key, 1, entry.key.length, ngrams, lmOrder);
				if (val.backoff != 0.0f)
					out.printf("%f\t%s\t%f\n", (float) log(prob), ngramString, (float) log(val.backoff));
				else
					out.printf("%f\t%s\n", (float) log(prob), ngramString);
			}
			out.println();

		}
	}

	/**
	 * @param prob
	 * @return
	 */
	private static float log(float prob) {
		return (float) (Math.log(prob) / Math.log(10.0));
	}

	//	from http://www-speech.sri.com/projects/srilm/manpages/ngram-discount.7.html

	//	p(a_z) = g(a_z) + bow(a_) p(_z)  ; Eqn.4
	//
	//	Let Z1 be the set {z: c(a_z) > 0}. For highest order N-grams we have:
	//
	//		g(a_z)  = max(0, c(a_z) - D) / c(a_)
	//		bow(a_) = 1 - Sum_Z1 g(a_z)
	//		        = 1 - Sum_Z1 c(a_z) / c(a_) + Sum_Z1 D / c(a_)
	//		        = D n(a_*) / c(a_)
	//
	//	Let Z2 be the set {z: n(*_z) > 0}. For lower order N-grams we have:
	//
	//		g(_z)  = max(0, n(*_z) - D) / n(*_*)
	//		bow(_) = 1 - Sum_Z2 g(_z)
	//		       = 1 - Sum_Z2 n(*_z) / n(*_*) + Sum_Z2 D / n(*_*)
	//		       = D n(_*) / n(*_*)
	//
	//	The original Kneser-Ney discounting (-ukndiscount) uses one discounting constant for each N-gram order. These constants are estimated as
	//
	//		D = n1 / (n1 + 2*n2)
	//
	//	where n1 and n2 are the total number of N-grams with exactly one and two counts, respectively. 

	private static float recurse(int[] ngram, int startPos, int endPos, HashNgramMap<KneserNeyCounts> ngrams, int lmOrder) {
		if (startPos == endPos) return 0.0f;
		ProbBackoffPair backoff = getLowerOrderProb(ngram, startPos, endPos - 1, ngrams, lmOrder);
		ProbBackoffPair prob = getLowerOrderProb(ngram, startPos, endPos, ngrams, lmOrder);

		return prob.prob + backoff.backoff * recurse(ngram, startPos + 1, endPos, ngrams, lmOrder);
	}

	private static ProbBackoffPair getHighestOrderProb(int[] key, KneserNeyCounts value, HashNgramMap<KneserNeyCounts> ngrams) {
		KneserNeyCounts rightDotCounts = getCounts(key, ngrams, 0, key.length - 1);
		float prob = Math.max(0.0f, value.tokenCounts - D) / rightDotCounts.tokenCounts;
		return new ProbBackoffPair(prob, 1.0f);
	}

	private static ProbBackoffPair getLowerOrderProb(int[] ngram, int startPos, int endPos, HashNgramMap<KneserNeyCounts> ngrams, int lmOrder) {
		if (startPos == endPos) return new ProbBackoffPair(1.0f, 1.0f);
		KneserNeyCounts counts = getCounts(ngram, ngrams, startPos, endPos);
		KneserNeyCounts prefixCounts = getCounts(ngram, ngrams, startPos, endPos - 1);
		final float d = ((endPos - startPos == 1) ? 0.0f : D);
		float prob = Math.max(0.0f, counts.leftDotTypeCounts - d) / prefixCounts.dotdotTypeCounts;
		float backoff = D * counts.rightDotTypeCounts / (endPos - startPos == lmOrder - 1 ? counts.tokenCounts : counts.dotdotTypeCounts);
		return new ProbBackoffPair((prob), (backoff));
	}

	/**
	 * @param key
	 * @param ngrams
	 * @param startPos
	 * @param endPos
	 */
	private static KneserNeyCounts getCounts(int[] key, HashNgramMap<KneserNeyCounts> ngrams, int startPos, int endPos) {
		final KneserNeyCounts value = new KneserNeyCounts();
		if (startPos > endPos) {
			@SuppressWarnings("unused")
			int x = 5;
		}
		if (startPos == endPos) {
			//only happens when requesting number of bigrams
			value.dotdotTypeCounts = (int) ngrams.getNumNgrams(1);
			return value;
		}
		LmContextInfo middleWords = ngrams.getOffsetForNgram(key, startPos, endPos);
		ngrams.getValues().getFromOffset(middleWords.offset, middleWords.order, value);
		return value;
	}

	/**
	 * @param ngrams
	 * @param lmOrder
	 * @param out
	 */
	private static void writeHeader(HashNgramMap<KneserNeyCounts> ngrams, int lmOrder, PrintWriter out) {
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			long numNgrams = ngrams.getNumNgrams(ngramOrder);
			out.println("ngram " + (ngramOrder + 1) + "=" + numNgrams);

		}
		out.println();
	}

}
