package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.PrintWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import edu.berkeley.nlp.lm.ArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Counter;
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
public class KneserNeyLmReaderCallback<W> implements NgramOrderedLmReaderCallback<LongRef>, LmReader<ProbBackoffPair, ArpaLmReaderCallback<ProbBackoffPair>>,
	ArrayEncodedNgramLanguageModel<W>, Serializable
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

	/**
	 * 
	 */
	protected static final long serialVersionUID = 1L;

	protected static final float DEFAULT_DISCOUNT = 0.75f;

	protected final int lmOrder;

	/**
	 * 
	 */

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

	protected final int startIndex;

	/**
	 * 
	 * @param wordIndexer
	 * @param maxOrder
	 * @param inputIsSentences
	 *            If true, input n-grams are assumed to be sentences, and all
	 *            sub-ngrams of up to order <code>maxOrder</code> are added. If
	 *            false, input n-grams are assumed to be atomic.
	 */
	public KneserNeyLmReaderCallback(final WordIndexer<W> wordIndexer, final int maxOrder) {
		this(wordIndexer, maxOrder, new ConfigOptions());
	}

	public KneserNeyLmReaderCallback(final WordIndexer<W> wordIndexer, final int maxOrder, final ConfigOptions opts) {
		this.lmOrder = maxOrder;
		this.startIndex = wordIndexer.getIndexPossiblyUnk(wordIndexer.getStartSymbol());

		this.opts = opts;
		double last = Double.NEGATIVE_INFINITY;
		for (final double c : opts.kneserNeyMinCounts) {
			if (c < last)
				throw new IllegalArgumentException("Please ensure that ConfigOptions.kneserNeyMinCounts is monotonic (value was "
					+ Arrays.toString(opts.kneserNeyMinCounts) + ")");
			last = c;
		}
		this.wordIndexer = wordIndexer;
		final KneserNeyCountValueContainer values = new KneserNeyCountValueContainer(lmOrder, startIndex);//, justLastWord);
		ngrams = HashNgramMap.createExplicitWordHashNgramMap(values, opts, lmOrder, false);

	}

	public void call(final W[] ngram, final LongRef value) {
		final int[] ints = new int[ngram.length];
		for (int i = 0; i < ngram.length; ++i)
			ints[i] = wordIndexer.getOrAddIndex(ngram[i]);
		call(ints, 0, ints.length, value, "");
	}

	public void callJustLast(final W[] ngram, final LongRef value, final long[][] scratch) {
		final int[] ints = new int[ngram.length];
		for (int i = 0; i < ngram.length; ++i)
			ints[i] = wordIndexer.getOrAddIndex(ngram[i]);
		addNgram(ints, 0, ints.length, value, "", true, scratch);
	}

	@Override
	public void call(final int[] ngram, final int startPos, final int endPos, final LongRef value, final String words) {
		final long[][] prevOffsets = new long[lmOrder][endPos - startPos];
		addNgram(ngram, startPos, endPos, value, words, false, prevOffsets);
	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param value
	 * @param words
	 */
	public void addNgram(final int[] ngram, final int startPos, final int endPos, final LongRef value, @SuppressWarnings("unused") final String words, final boolean justLastWord,
		final long[][] scratch) {

		final KneserNeyCounts scratchCounts = new KneserNeyCounts();
		ngrams.rehashIfNecessary(endPos - startPos);
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			for (int i = startPos; i < endPos; ++i) {
				int j = i + ngramOrder + 1;
				if (j > endPos) continue;
				scratchCounts.tokenCounts = value.value;
				final long prevOffset = ngramOrder == 0 ? 0 : scratch[ngramOrder - 1][i];
				final long suffixOffset = ngramOrder == 0 ? 0 : scratch[ngramOrder - 1][i + 1];
				assert prevOffset >= 0;
				scratch[ngramOrder][i - startPos] = ngrams.putWithOffsetAndSuffix(ngram, i, j, prevOffset, suffixOffset, !justLastWord || j == endPos
				/* || ngram[startPos] == startIndex */
				? scratchCounts : null);
			}
		}

	}

	protected float interpolateProb(final int[] ngram, final int startPos, final int endPos) {
		if (startPos == endPos) return 0.0f;
		final float backoff = getLowerOrderBackoff(ngram, startPos, endPos - 1);
		final float prob = getLowerOrderProb(ngram, startPos, endPos);
		return prob + backoff * interpolateProb(ngram, startPos + 1, endPos);
	}

	protected float getHighestOrderProb(final int[] ngram, final int startPos, final int endPos) {
		final KneserNeyCounts counts = getCounts(ngram, startPos, endPos, false);
		final KneserNeyCounts rightDotCounts = getCounts(ngram, startPos, endPos - 1, true);
		final int ngramOrder = endPos - startPos - 1;
		final float D = getDiscountForOrder(ngramOrder);
		final float prob = rightDotCounts.tokenCounts == 0 ? 0.0f : Math.max(0.0f, (counts.tokenCounts - D) / rightDotCounts.tokenCounts);
		return prob;
	}

	protected float getLowerOrderProb(final int[] ngram, final int startPos, final int endPos) {
		if (startPos == endPos) return 1.0f;
		final KneserNeyCounts counts = getCounts(ngram, startPos, endPos, false);
		final KneserNeyCounts prefixCounts = getCounts(ngram, startPos, endPos - 1, true);

		final float probDiscount = (endPos - startPos == 1) ? 0.0f : getDiscountForOrder(endPos - startPos - 1);
		final float prob = prefixCounts.dotdotTypeCounts == 0 ? 0.0f : Math.max(0.0f, counts.leftDotTypeCounts - probDiscount) / prefixCounts.dotdotTypeCounts;

		return prob;
	}

	protected float getLowerOrderBackoff(final int[] ngram, final int startPos, final int endPos) {
		if (startPos == endPos) return 1.0f;
		final KneserNeyCounts counts = getCounts(ngram, startPos, endPos, true);
		final long backoffDenom = (endPos - startPos == lmOrder - 1 || ngram[startPos] == startIndex) ? counts.tokenCounts : counts.dotdotTypeCounts;

		assert backoffDenom >= 0;
		//		final long backoffDenom = endPos - startPos == lmOrder - 1 ? counts.tokenCounts : counts.dotdotTypeCounts;
		final float backoffDiscount = getDiscountForOrder(endPos - startPos);
		final float backoff = backoffDenom == 0.0f ? 1.0f : backoffDiscount * counts.rightDotTypeCounts / backoffDenom;
		return backoff;
	}

	protected float getDiscountForOrder(int ngramOrder) {
		if (opts.kneserNeyDiscounts != null) return (float) opts.kneserNeyDiscounts[ngramOrder];
		final int numOneCounters = ((KneserNeyCountValueContainer) ngrams.getValues()).getNumOneCountNgrams(ngramOrder);
		final int numTwoCounters = ((KneserNeyCountValueContainer) ngrams.getValues()).getNumTwoCountNgrams(ngramOrder);
		final float denom = (numOneCounters + 2 * (float) numTwoCounters);
		return denom == 0.0f ? 1e-5f : numOneCounters / denom;
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
	private KneserNeyCounts getCounts(final int[] key, final int startPos, final int endPos, final boolean isBackoff) {
		final KneserNeyCounts value = new KneserNeyCounts();
		if (startPos == endPos) {
			//only happens when requesting number of bigrams
			value.dotdotTypeCounts = ((KneserNeyCountValueContainer) ngrams.getValues()).getBigramTypeCounts();
			return value;
		}
		final long offset = ngrams.getOffsetForNgramInModel(key, startPos, endPos);
		if (offset < 0) return value;
		ngrams.getValues().getFromOffset(offset, endPos - startPos - 1, value);

		final boolean startsWithStartSym = key[startPos] == startIndex;
		final boolean endsWithEndSym = key[endPos - 1] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getEndSymbol());
		if (startsWithStartSym) {
			value.dotdotTypeCounts = value.rightDotTypeCounts;
			if (endPos - startPos < lmOrder - 1 || (endPos - startPos == lmOrder - 1 && !isBackoff)) value.tokenCounts = value.leftDotTypeCounts;
		}
		if (endsWithEndSym) {
			value.rightDotTypeCounts = 1;
			value.dotdotTypeCounts = value.leftDotTypeCounts;
		}
		return value;
	}

	public static double[] defaultDiscounts() {
		return constantArray(defaultMinCounts().length, DEFAULT_DISCOUNT);
	}

	public static double[] defaultMinCounts() {
		//same as SRILM
		return new double[] { 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2 };
	}

	private static double[] constantArray(final int n, final double f) {
		final double[] ret = new double[n];
		Arrays.fill(ret, f);
		return ret;
	}

	@Override
	public void parse(ArpaLmReaderCallback<ProbBackoffPair> callback) {
		Logger.startTrack("Writing Kneser-Ney probabilities");

		List<Long> lengths = new ArrayList<Long>();
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			Logger.startTrack("Counting counts for order " + ngramOrder);
			long numNgrams = 0; //ngrams.getNumNgrams(ngramOrder);
			for (final Entry<KneserNeyCounts> entry : ngrams.getNgramsForOrder(ngramOrder)) {
				final long relevantCount = entry.value.tokenCounts;
				if (ngramOrder >= lmOrder - 2 && ngramOrder < opts.kneserNeyMinCounts.length && relevantCount < opts.kneserNeyMinCounts[ngramOrder]) continue;
				numNgrams++;
			}
			lengths.add(numNgrams);
			Logger.endTrack();
		}
		callback.initWithLengths(lengths);
		for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
			callback.handleNgramOrderStarted(ngramOrder + 1);

			Logger.logss("On order " + (ngramOrder + 1));
			int linenum = 0;
			for (final Entry<KneserNeyCounts> entry : ngrams.getNgramsForOrder(ngramOrder)) {
				if (linenum++ % 10000 == 0) Logger.logs("Writing line " + linenum);
				final long relevantCount = entry.value.tokenCounts;
				if (ngramOrder >= lmOrder - 2 && ngramOrder < opts.kneserNeyMinCounts.length && relevantCount < opts.kneserNeyMinCounts[ngramOrder]) continue;

				final int[] ngram = entry.key;
				final int endPos = ngram.length;
				final int startPos = 0;
				ProbBackoffPair value = getProbBackoff(ngram, startPos, endPos);
				callback.call(ngram, startPos, endPos, value, "");

			}
			callback.handleNgramOrderFinished(ngramOrder + 1);

		}
		callback.cleanup();

		Logger.endTrack();
	}

	/**
	 * @param startIndex
	 * @param ngramOrder
	 * @param entry
	 * @param ngram
	 * @param endPos
	 * @param startPos
	 * @return
	 */
	private ProbBackoffPair getProbBackoff(final int[] ngram, final int startPos, final int endPos) {
		final int ngramOrder = endPos - startPos - 1;
		final boolean isHighestOrder = ngramOrder == lmOrder - 1;
		final float val = isHighestOrder || ngram[startPos] == startIndex ? getHighestOrderProb(ngram, startPos, endPos) : getLowerOrderProb(ngram, startPos,
			endPos);
		int nextNonStart = startPos + 1;
		while (nextNonStart < endPos && ngram[nextNonStart] == startIndex) {
			nextNonStart++;
		}
		final float prob = val + getLowerOrderBackoff(ngram, startPos, endPos - 1) * interpolateProb(ngram, nextNonStart, endPos);
		final boolean isStartEndSym = endPos - startPos == 1 && ngram[startPos] == startIndex;
		final float logProb = isStartEndSym ? -99 : ((float) (Math.log10(prob)));
		//		if (logProb == Float.NEGATIVE_INFINITY) {
		//			System.out.println("here");
		//		}
		final float backoff = isHighestOrder ? 0.0f : (float) Math.log10(getLowerOrderBackoff(ngram, startPos, endPos));
		final ProbBackoffPair ret = new ProbBackoffPair(logProb, backoff);
		return ret;
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

	@Override
	public int getLmOrder() {
		return lmOrder;
	}

	@Override
	public float scoreSentence(List<W> sentence) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.scoreSentence(sentence, this);
	}

	@Override
	public float getLogProb(List<W> ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	@Override
	public float getLogProb(int[] ngram, int startPos, int endPos) {
		ProbBackoffPair probBackoff = getProbBackoff(ngram, startPos, endPos);
		return probBackoff.prob;
	}

	@Override
	public float getLogProb(int[] ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	public long getTotalSize() {
		return ngrams.getTotalSize();
	}

	@Override
	public void setOovWordLogProb(float logProb) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

}
