package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap.Entry;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.StrUtils;
import edu.berkeley.nlp.lm.values.KneserNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.KneserNeyCountValueContainer.KneserNeyCounts;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

/**
 * Class for producing a Kneser-Ney language model in ARPA format from raw text.
 * 
 * Confusingly, this class is both a {@link LmReaderCallback} (called from
 * {@link TextReader}, which reads plain text), and a {@link LmReader}, which
 * "reads" counts and produces Kneser-Ney probabilities and backoffs and passes
 * them on an {@link ArpaLmReaderCallback}
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class KneserNeyLmReaderCallback<W> implements NgramOrderedLmReaderCallback<LongRef>, LmReader<ProbBackoffPair, ArpaLmReaderCallback<ProbBackoffPair>>
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

	private static final int MAX_ORDER = 10;

	private static final float DEFAULT_DISCOUNT = 0.75f;

	protected final int lmOrder;

	/**
	 * 
	 * This array represents the discount used for each ngram order.
	 * 
	 * The original Kneser-Ney discounting (-ukndiscount) uses one discounting
	 * constant for each N-gram order. These constants are estimated as
	 * 
	 * D = n1 / (n1 + 2*n2)
	 * 
	 * where n1 and n2 are the total number of N-grams with exactly one and two
	 * counts, respectively.
	 * 
	 * For simplicity, our code just uses a constant discount for each order of
	 * 0.75. However, other discounts can be specified.
	 */

	protected final WordIndexer<W> wordIndexer;

	protected final HashNgramMap<KneserNeyCounts> ngrams;

	protected final ConfigOptions opts;

	private boolean inputIsSentences;

	/**
	 * 
	 * @param wordIndexer
	 * @param maxOrder
	 * @param inputIsSentences
	 *            If true, input n-grams are assumed to be sentences, and all
	 *            sub-ngrams of up to order <code>maxOrder</code> are added. If
	 *            false, input n-grams are assumed to be atomic.
	 */
	public KneserNeyLmReaderCallback(final WordIndexer<W> wordIndexer, final int maxOrder, final boolean inputIsSentences) {
		this(wordIndexer, maxOrder, inputIsSentences, new ConfigOptions());
	}

	public KneserNeyLmReaderCallback(final WordIndexer<W> wordIndexer, final int maxOrder, final boolean inputIsSentences, final ConfigOptions opts) {
		this.lmOrder = maxOrder;
		this.inputIsSentences = inputIsSentences;

		if (maxOrder >= MAX_ORDER) throw new IllegalArgumentException("Reguested n-grams of order " + maxOrder + " but we only allow up to " + 10);
		this.opts = opts;
		final double last = Double.NEGATIVE_INFINITY;
		for (final double c : opts.kneserNeyMinCounts) {
			if (c < last)
				throw new IllegalArgumentException("Please ensure that ConfigOptions.kneserNeyMinCounts is monotonic (value was "
					+ Arrays.toString(opts.kneserNeyMinCounts) + ")");
		}
		this.wordIndexer = wordIndexer;
		final KneserNeyCountValueContainer values = new KneserNeyCountValueContainer(lmOrder);
		ngrams = HashNgramMap.createExplicitWordHashNgramMap(values, new ConfigOptions(), lmOrder, false);

	}

	public void call(final W[] ngram) {
		final int[] ints = new int[ngram.length];
		for (int i = 0; i < ngram.length; ++i)
			ints[i] = wordIndexer.getOrAddIndex(ngram[i]);
		call(ints, 0, ints.length, new LongRef(1), "");
	}

	@Override
	public void call(final int[] ngram, final int startPos, final int endPos, final LongRef value, final String words) {
		if (inputIsSentences) {
			final long[][] prevOffsets = new long[lmOrder][endPos - startPos];
			for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
				for (int i = startPos; i < endPos; ++i) {
					int j = i + ngramOrder + 1;
					if (j > endPos) continue;
					final KneserNeyCounts counts = new KneserNeyCounts();
					counts.tokenCounts = value.value;
					final long prevOffset = ngramOrder == 0 ? 0 : prevOffsets[ngramOrder - 1][i];
					prevOffsets[ngramOrder][i - startPos] = ngrams.putWithOffset(ngram, i, j, prevOffset, counts);
				}
			}
			ngrams.rehashIfNecessary();
		} else {
			final KneserNeyCounts counts = new KneserNeyCounts();
			counts.tokenCounts = value.value;
			final long add = ngrams.put(ngram, startPos, endPos, counts);
			if (add < 0) { throw new RuntimeException("Failed to add line " + words); }

		}
	}

	protected float interpolateProb(final int[] ngram, final int startPos, final int endPos) {
		if (startPos == endPos) return 0.0f;
		final ProbBackoffPair backoff = getLowerOrderProb(ngram, startPos, endPos - 1);
		final ProbBackoffPair prob = getLowerOrderProb(ngram, startPos, endPos);
		return prob.prob + backoff.backoff * interpolateProb(ngram, startPos + 1, endPos);
	}

	protected ProbBackoffPair getHighestOrderProb(final int[] key, final KneserNeyCounts value) {
		final KneserNeyCounts rightDotCounts = getCounts(key, 0, key.length - 1);
		final float D = (float) opts.kneserNeyDiscounts[key.length - 1];
		final float prob = Math.max(0.0f, value.tokenCounts - D) / rightDotCounts.tokenCounts;
		return new ProbBackoffPair(prob, 1.0f);
	}

	protected ProbBackoffPair getLowerOrderProb(final int[] ngram, final int startPos, final int endPos) {
		if (startPos == endPos) return new ProbBackoffPair(1.0f, 1.0f);
		final KneserNeyCounts counts = getCounts(ngram, startPos, endPos);
		final KneserNeyCounts prefixCounts = getCounts(ngram, startPos, endPos - 1);

		final float probDiscount = ((endPos - startPos == 1) ? 0.0f : (float) opts.kneserNeyDiscounts[endPos - startPos - 1]);
		final float prob = Math.max(0.0f, counts.leftDotTypeCounts - probDiscount) / prefixCounts.dotdotTypeCounts;

		final long backoffDenom = endPos - startPos == lmOrder - 1 ? counts.tokenCounts : counts.dotdotTypeCounts;
		final float backoffDiscount = (float) opts.kneserNeyDiscounts[endPos - startPos];
		final float backoff = backoffDenom == 0.0f ? 1.0f : backoffDiscount * counts.rightDotTypeCounts / backoffDenom;
		return new ProbBackoffPair((prob), (backoff));
	}

	@Override
	public void cleanup() {

	}

	/**
	 * @param key
	 * @param ngrams
	 * @param startPos
	 * @param endPos
	 */
	private KneserNeyCounts getCounts(final int[] key, final int startPos, final int endPos) {
		final KneserNeyCounts value = new KneserNeyCounts();
		if (startPos == endPos) {
			//only happens when requesting number of bigrams
			value.dotdotTypeCounts = (int) ngrams.getNumNgrams(1);
			return value;
		}
		final LmContextInfo middleWords = ngrams.getOffsetForNgram(key, startPos, endPos);
		ngrams.getValues().getFromOffset(middleWords.offset, middleWords.order, value);
		final boolean startsWithStartSym = key[startPos] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getStartSymbol());
		final boolean endsWithEndSym = key[endPos - 1] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getEndSymbol());
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

	public static double[] defaultDiscounts() {
		return constantArray(MAX_ORDER, DEFAULT_DISCOUNT);
	}

	public static double[] defaultMinCounts() {
		//same as SRILM
		return new double[] { 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2 };
	}

	private static double[] constantArray(final int n, final double f) {
		final double[] ret = new double[n];
		Arrays.fill(ret, f);
		return ret;
	}

	@Override
	public void parse(ArpaLmReaderCallback<ProbBackoffPair> callback) {
		Logger.startTrack("Writing Kneser-Ney probabilities");
		final int startIndex = wordIndexer.getIndexPossiblyUnk(wordIndexer.getStartSymbol());
		List<Long> lengths = new ArrayList<Long>();
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			final long numNgrams = ngrams.getNumNgrams(ngramOrder);
			lengths.add(numNgrams);
		}
		callback.initWithLengths(lengths);
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			callback.handleNgramOrderStarted(ngramOrder + 1);

			Logger.logss("On order " + (ngramOrder + 1));
			int linenum = 0;
			for (final Entry<KneserNeyCounts> entry : ngrams.getNgramsForOrder(ngramOrder)) {
				if (linenum++ % 10000 == 0) Logger.logs("Writing line " + linenum);
				if (ngramOrder >= lmOrder - 2 && entry.value.tokenCounts < opts.kneserNeyMinCounts[ngramOrder]) continue;

				final ProbBackoffPair val = ngramOrder == lmOrder - 1 ? getHighestOrderProb(entry.key, entry.value) : getLowerOrderProb(entry.key, 0,
					entry.key.length);
				final float prob = val.prob + getLowerOrderProb(entry.key, 0, entry.key.length - 1).backoff * interpolateProb(entry.key, 1, entry.key.length);
				final boolean isStartEndSym = entry.key.length == 1 && entry.key[0] == startIndex;
				final float logProb = isStartEndSym ? -99 : ((float) (Math.log10(prob)));
				final float backoff = (float) Math.log10(val.backoff);
				final int[] ngram = entry.key;
				callback.call(ngram, 0, ngram.length, new ProbBackoffPair(logProb, backoff), "");

			}
			callback.handleNgramOrderFinished(ngramOrder + 1);

		}
		callback.cleanup();

		Logger.endTrack();
	}

	public WordIndexer<W> getWordIndexer() {
		return wordIndexer;
	}

	@Override
	public void handleNgramOrderFinished(int order) {
	}

	@Override
	public void handleNgramOrderStarted(int order) {
	}

}
