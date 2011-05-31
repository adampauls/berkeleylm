package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap.Entry;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.StrUtils;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

/**
 * Class for producing a Kneser-Ney language model in ARPA format from raw text. 
 * @author adampauls
 *
 * @param <W>
 */
public class KneserNeyFromTextReader<W>
{

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

	private static final float DEFAULT_DISCOUNT = 0.75f;

	private int lmOrder;

	private float[] discounts;

	private WordIndexer<W> wordIndexer;


	public KneserNeyFromTextReader(WordIndexer<W> wordIndexer, int maxOrder) {
		this(wordIndexer, maxOrder, constantArray(maxOrder, DEFAULT_DISCOUNT));
	}

	public KneserNeyFromTextReader(WordIndexer<W> wordIndexer, int maxOrder, float[] discounts) {
		this.lmOrder = maxOrder;
		this.discounts = discounts;
		this.wordIndexer = wordIndexer;

	}

	/**
	 * Reads newline-separated plain text from inputFiles, and writes an ARPA lm
	 * file to outputFile. If files have a .gz suffix, then they will be
	 * (un)zipped as necessary.
	 * 
	 * @param inputFiles
	 * @param outputFile
	 */
	public void readFromFiles(List<File> inputFiles, File outputFile) {
		readFromFiles(inputFiles, IOUtils.openOutHard(outputFile));
	}

	void readFromFiles(List<File> inputFiles, PrintWriter outputFile) {
		HashNgramMap<KneserNeyCounts> readFromFiles = readFromFiles(inputFiles);
		writeToPrintWriter(readFromFiles, outputFile);
	}

	private HashNgramMap<KneserNeyCounts> readFromFiles(Iterable<File> files) {
		Logger.startTrack("Reading from files " + files);
		final Iterable<String> allLinesIterator = getLineIterator(files);

		final HashNgramMap<KneserNeyCounts> countNgrams = countNgrams(allLinesIterator);
		Logger.endTrack();
		return countNgrams;

	}

	/**
	 * @param <W>
	 * @param wordIndexer
	 * @param maxOrder
	 * @param allLinesIterator
	 * @param ngrams
	 * @return
	 */
	private HashNgramMap<KneserNeyCounts> countNgrams(final Iterable<String> allLinesIterator) {
		final KneseryNeyCountValueContainer values = new KneseryNeyCountValueContainer(lmOrder);
		HashNgramMap<KneserNeyCounts> ngrams = HashNgramMap.createExplicitWordHashNgramMap(values, new ConfigOptions(), lmOrder, false);
		long numLines = 0;
	
		for (String line : allLinesIterator) {
			if (numLines % 10000 == 0) Logger.logs("On line " + numLines);
			numLines++;
			final String[] words = line.split(" ");
			int[] sent = new int[words.length + 2];
			sent[0] = wordIndexer.getOrAddIndex(wordIndexer.getStartSymbol());
			sent[sent.length - 1] = wordIndexer.getOrAddIndex(wordIndexer.getEndSymbol());
			for (int i = 0; i < words.length; ++i) {
				sent[i + 1] = wordIndexer.getOrAddIndexFromString(words[i]);
			}
			for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
				for (int i = 0; i < sent.length; ++i) {
					if (i - ngramOrder < 0) continue;
					ngrams.put(sent, i - ngramOrder, i + 1, null);
				}
			}
		}
		return ngrams;
	}

	/**
	 * @param files
	 * @return
	 */
	private Iterable<String> getLineIterator(Iterable<File> files) {
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
		return allLinesIterator;
	}

	/**
	 * @param <W>
	 * @param wordIndexer
	 * @param ngrams
	 * @param out
	 */
	void writeToPrintWriter(HashNgramMap<KneserNeyCounts> ngrams, PrintWriter out) {
		out.println();
		out.println("\\data\\");
		writeHeader(ngrams, lmOrder, out);
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			out.println("\\" + (ngramOrder + 1) + "-grams:");
			for (Entry<KneserNeyCounts> entry : ngrams.getNgramsForOrder(ngramOrder)) {
				final String ngramString = StrUtils.join(WordIndexer.StaticMethods.toList(wordIndexer, entry.key));
				if (ngramString.startsWith("<s> This is")) {
					@SuppressWarnings("unused")
					int x = 5;
				}
				ProbBackoffPair val = ngramOrder == lmOrder - 1 ? getHighestOrderProb(entry.key, entry.value, ngrams) : getLowerOrderProb(entry.key, 0,
					entry.key.length, ngrams);
				float prob = val.prob + getLowerOrderProb(entry.key, 0, entry.key.length - 1, ngrams).backoff * interpolateProb(entry.key, 1, entry.key.length, ngrams);
				boolean endsWithEndSym = entry.key[entry.key.length - 1] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getEndSymbol());
				boolean isStartEndSym = entry.key.length == 1 && entry.key[0] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getStartSymbol());
				final float logProb = isStartEndSym ? -99 : log(prob);
				if (endsWithEndSym || val.backoff == 1.0f)
					out.printf("%f\t%s\n", logProb, ngramString);
				else
					out.printf("%f\t%s\t%f\n", logProb, ngramString, log(val.backoff));
			}
			out.println();

		}
		out.println("\\end\\");
		out.close();
	}

	private float interpolateProb(int[] ngram, int startPos, int endPos, HashNgramMap<KneserNeyCounts> ngrams) {
		if (startPos == endPos) return 0.0f;
		ProbBackoffPair backoff = getLowerOrderProb(ngram, startPos, endPos - 1, ngrams);
		ProbBackoffPair prob = getLowerOrderProb(ngram, startPos, endPos, ngrams);
		return prob.prob + backoff.backoff * interpolateProb(ngram, startPos + 1, endPos, ngrams);
	}

	private ProbBackoffPair getHighestOrderProb(int[] key, KneserNeyCounts value, HashNgramMap<KneserNeyCounts> ngrams) {
		KneserNeyCounts rightDotCounts = getCounts(key, ngrams, 0, key.length - 1);
		final float D = discounts[key.length - 1];
		float prob = Math.max(0.0f, value.tokenCounts - D) / rightDotCounts.tokenCounts;
		return new ProbBackoffPair(prob, 1.0f);
	}

	private ProbBackoffPair getLowerOrderProb(int[] ngram, int startPos, int endPos, HashNgramMap<KneserNeyCounts> ngrams) {
		if (startPos == endPos) return new ProbBackoffPair(1.0f, 1.0f);
		KneserNeyCounts counts = getCounts(ngram, ngrams, startPos, endPos);
		KneserNeyCounts prefixCounts = getCounts(ngram, ngrams, startPos, endPos - 1);
		
		final float probDiscount = ((endPos - startPos == 1) ? 0.0f : discounts[endPos - startPos - 1]);
		float prob = Math.max(0.0f, counts.leftDotTypeCounts - probDiscount) / prefixCounts.dotdotTypeCounts;
		
		final long backoffDenom = endPos - startPos == lmOrder - 1 ? counts.tokenCounts : counts.dotdotTypeCounts;
		final float backoffDiscount = discounts[endPos - startPos];
		float backoff = backoffDenom == 0.0f ? 1.0f : backoffDiscount * counts.rightDotTypeCounts / backoffDenom;
		return new ProbBackoffPair((prob), (backoff));
	}

	/**
	 * @param key
	 * @param ngrams
	 * @param startPos
	 * @param endPos
	 */
	private KneserNeyCounts getCounts(int[] key, HashNgramMap<KneserNeyCounts> ngrams, int startPos, int endPos) {
		final KneserNeyCounts value = new KneserNeyCounts();
		if (startPos == endPos) {
			//only happens when requesting number of bigrams
			value.dotdotTypeCounts = (int) ngrams.getNumNgrams(1);
			return value;
		}
		LmContextInfo middleWords = ngrams.getOffsetForNgram(key, startPos, endPos);
		ngrams.getValues().getFromOffset(middleWords.offset, middleWords.order, value);
		boolean startsWithStartSym = key[startPos] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getStartSymbol());
		boolean endsWithEndSym = key[endPos - 1] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getEndSymbol());
		if (startsWithStartSym) {
			value.leftDotTypeCounts = 1;
			value.dotdotTypeCounts = value.rightDotTypeCounts;
		}
		if (endsWithEndSym) {
			value.rightDotTypeCounts = 1;
			value.dotdotTypeCounts = value.leftDotTypeCounts;
		}
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

	private static float[] constantArray(int n, float f) {
		float[] ret = new float[n];
		Arrays.fill(ret, f);
		return ret;
	}

	/**
	 * @param prob
	 * @return
	 */
	private static float log(float prob) {
		return (float) (Math.log(prob) / Math.log(10.0));
	}

}
