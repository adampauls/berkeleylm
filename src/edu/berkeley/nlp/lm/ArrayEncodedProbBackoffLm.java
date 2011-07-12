package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

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

	public ArrayEncodedProbBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final NgramMap<ProbBackoffPair> map, final ConfigOptions opts) {
		super(lmOrder, wordIndexer, (float) opts.unknownWordLogProb);
		this.map = map;
		this.values = (ProbBackoffValueContainer) map.getValues();

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
		final NgramMap<ProbBackoffPair> localMap = map;
		final ContextEncodedNgramMap<ProbBackoffPair> contextEncodedLocalMap = map instanceof ContextEncodedNgramMap ? (ContextEncodedNgramMap<ProbBackoffPair>) map
			: null;
		float backoff = 0.0f;

		long probContext = 0L;
		int probContextOrder = -1;
		long matchedProbContext = -1L;
		int matchedProbContextOrder = -1;

		final ProbBackoffPair scratch = contextEncodedLocalMap != null ? null : new ProbBackoffPair(Float.NaN, Float.NaN);
		for (int i = endPos - 1; i >= startPos; --i) {
			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);
			if (probContext < 0) break;

			matchedProbContext = probContext;
			matchedProbContextOrder = probContextOrder;
			probContextOrder++;
		}
		if (matchedProbContext < 0) return oovWordLogProb;
		float logProb = scratch == null ? values.getProb(matchedProbContextOrder + 1, matchedProbContext) : scratch.prob;
		if (Float.isNaN(logProb)) {
			// this was a fake entry, let's do it again, but only keep track of the biggest match which was not fake
			probContext = 0L;
			probContextOrder = -1;
			for (int i = endPos - 1; i >= startPos; --i) {
				probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);
				if (probContext < 0) break;
				float tmpProb = scratch == null ? values.getProb(probContextOrder + 1, probContext) : scratch.prob;
				if (!Float.isNaN(tmpProb)) {
					logProb = tmpProb;
					matchedProbContext = probContext;
					matchedProbContextOrder = probContextOrder;
				}
				probContextOrder++;
			}
		}

		// matched the whole n-gram, so no need to back off
		if (matchedProbContextOrder == endPos - startPos - 2) return logProb;
		long backoffContext = 0L;
		int backoffContextOrder = -1;
		for (int i = 0; i < endPos - startPos - 1; ++i) {
			backoffContext = localMap.getValueAndOffset(backoffContext, backoffContextOrder, ngram[endPos - i - 2], scratch);
			if (backoffContext < 0) break;
			backoffContextOrder++;
			if (i > matchedProbContextOrder) {
				final float currBackoff = scratch == null ? values.getBackoff(backoffContextOrder, backoffContext) : scratch.backoff;
				backoff += Float.isNaN(currBackoff) ? 0.0f : currBackoff;
			}
		}
		return logProb + backoff;
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
