package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

/**
 * Language model implementation which uses Kneser-Ney-style backoff
 * computation.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class ProbBackoffLm<W> extends AbstractArrayEncodedNgramLanguageModel<W> implements ArrayEncodedNgramLanguageModel<W>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final NgramMap<ProbBackoffPair> map;
	

	public ProbBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final NgramMap<ProbBackoffPair> map, final ConfigOptions opts) {
		super(lmOrder, wordIndexer, (float)opts.unknownWordLogProb);
		this.map = map;

	}

	/*
	 * (non-Javadoc)
	 * @see edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel#getLogProb(int[], int, int)
	 */
	@Override
	public float getLogProb(final int[] ngram, final int startPos_, final int endPos_) {
		final NgramMap<ProbBackoffPair> localMap = map;
		float logProb = oovWordLogProb;
		float backoff = 0.0f;

		long probContext = 0L;
		int probContextOrder = -1;

		long backoffContext = 0L;
		int backoffContextOrder = -1;
		final ProbBackoffPair scratch = new ProbBackoffPair(Float.NaN, Float.NaN);
		for (int i = endPos_ - 1; i >= startPos_; --i) {
			if (probContext >= 0) {
				probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);
			}
			if (probContext >= 0) {
				probContextOrder++;
				final float currProb = scratch.prob;
				if (Float.isNaN(currProb) && i == startPos_) {
					return logProb + backoff;
				} else if (!Float.isNaN(currProb)) {
					logProb = currProb;
					backoff = 0.0f;
				}
			} else {
				if (i == endPos_ - 1) return oovWordLogProb;
			}
			if (i == startPos_) break;

			backoffContext = localMap.getValueAndOffset(backoffContext, backoffContextOrder, ngram[i - 1], scratch);
			if (backoffContext < 0) break;
			backoffContextOrder++;
			final float currBackoff = scratch.backoff;
			backoff += Float.isNaN(currBackoff) ? 0.0f : currBackoff;
		}
		return logProb + backoff;
	}

	/*
	 * (non-Javadoc)
	 * @see edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel#getLogProb(int[])
	 */
	@Override
	public float getLogProb(final int[] ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	/*
	 * (non-Javadoc)
	 * @see edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel#getLogProb(java.util.List)
	 */
	@Override
	public float getLogProb(final List<W> ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

}
