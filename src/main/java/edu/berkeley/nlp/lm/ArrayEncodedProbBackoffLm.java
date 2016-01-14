package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.berkeley.nlp.lm.collections.Counter;
import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

/**
 * Language model implementation which uses Kneser-Ney-style backoff
 * computation.
 * 
 * Note that unlike the description in Pauls and Klein (2011), we store trie for
 * which the first word in n-gram points to its prefix for this particular
 * implementation. This is in contrast to {@link ContextEncodedProbBackoffLm},
 * which stores a trie for which the last word points to its suffix. This was
 * done because it simplifies the code significantly, without significantly
 * changing speed or memory usage.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class ArrayEncodedProbBackoffLm<W> extends AbstractArrayEncodedNgramLanguageModel<W> implements ArrayEncodedNgramLanguageModel<W>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final NgramMap<ProbBackoffPair> map;

	private final ProbBackoffValueContainer values;

	private final boolean useScratchValues;

	private final long numWords;

	public ArrayEncodedProbBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final NgramMap<ProbBackoffPair> map, final ConfigOptions opts) {
		super(lmOrder, wordIndexer, (float) opts.unknownWordLogProb);
		this.map = map;
		this.values = (ProbBackoffValueContainer) map.getValues();
		useScratchValues = !(map instanceof ContextEncodedNgramMap);
		numWords = map.getNumNgrams(0);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel#getLogProb
	 * (int[], int, int)
	 */
	@Override
	public float getLogProb(final int[] ngram, final int startPos, final int endPos) {
		int startPos_ = startPos;
		while (endPos - startPos_ > lmOrder) {
			startPos_++;	
		}
		final NgramMap<ProbBackoffPair> localMap = map;
		if (endPos - startPos_ < 1) return 0.0f;

		final ProbBackoffPair scratch = !useScratchValues ? null : new ProbBackoffPair(Float.NaN, Float.NaN);
		final int unigramWord = ngram[endPos - 1];
		if (unigramWord < 0 || unigramWord >= numWords) return oovWordLogProb;

		long matchedProbContext = unigramWord;
		int matchedProbContextOrder = -1;
		for (int i = endPos - 2; i >= startPos_; --i) {
			final int probContextOrder = endPos - i - 2;
			final long probContext = localMap.getValueAndOffset(matchedProbContext, probContextOrder, ngram[i], scratch);
			if (probContext < 0) break;
			matchedProbContext = probContext;
			matchedProbContextOrder = probContextOrder;
		}
		float logProb = scratch == null ? values.getProb(matchedProbContextOrder + 1, matchedProbContext) : scratch.prob;
		if (Float.isNaN(logProb)) {
			// this was a fake entry, let's do it again, but only keep track of the biggest match which was not fake
			matchedProbContext = 0;
			matchedProbContextOrder = -1;
			for (int i = endPos - 1; i >= startPos_; --i) {
				final int probContextOrder = endPos - i - 2;
				final long probContext = localMap.getValueAndOffset(matchedProbContext, probContextOrder, ngram[i], scratch);
				if (probContext < 0) break;
				final float tmpProb = scratch == null ? values.getProb(probContextOrder + 1, probContext) : scratch.prob;
				if (!Float.isNaN(tmpProb)) {
					logProb = tmpProb;
					matchedProbContext = probContext;
					matchedProbContextOrder = probContextOrder;
				}
			}
		}

		final float backoff = matchedProbContextOrder == endPos - startPos_ - 2 || endPos - startPos_ <= 1 ? 0.0f : getBackoffSum(ngram, startPos_, endPos,
			localMap, matchedProbContextOrder, scratch);
		return logProb + backoff;
	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param localMap
	 * @param matchedProbContextOrder
	 * @param scratch
	 * @return
	 */
	private float getBackoffSum(final int[] ngram, final int startPos, final int endPos, final NgramMap<ProbBackoffPair> localMap, int matchedProbContextOrder,
		final ProbBackoffPair scratch) {
		final long unigramWord = ngram[endPos - 2];
		if (unigramWord < 0 || unigramWord >= numWords) return 0.0f;
		long backoffContext = unigramWord;
		float backoff = 0.0f;

		// check if must include unigram backoff
		if (matchedProbContextOrder < 0) {
			if (scratch != null) {
				localMap.getValueAndOffset(0, -1, ngram[endPos - 2], scratch);
				backoff = scratch.backoff;
			} else {
				backoff = values.getBackoff(0, backoffContext);
			}
		}
		int i = 1;
		for (; i <= matchedProbContextOrder && backoffContext >= 0; ++i) {
			backoffContext = localMap.getValueAndOffset(backoffContext, i - 1, ngram[endPos - i - 2], null);
		}
		for (; i < endPos - startPos - 1 && backoffContext >= 0; ++i) {
			final int backoffContextOrder = i - 1;
			backoffContext = localMap.getValueAndOffset(backoffContext, backoffContextOrder, ngram[endPos - i - 2], scratch);
			if (backoffContext < 0) break;
			assert i > matchedProbContextOrder;
			final float currBackoff = scratch == null ? values.getBackoff(backoffContextOrder + 1, backoffContext) : scratch.backoff;
			backoff += Float.isNaN(currBackoff) ? 0.0f : currBackoff;
		}
		return backoff;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel#getLogProb
	 * (int[])
	 */
	@Override
	public float getLogProb(final int[] ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel#getLogProb
	 * (java.util.List)
	 */
	@Override
	public float getLogProb(final List<W> ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	public NgramMap<ProbBackoffPair> getNgramMap() {
		return map;
	}

}
