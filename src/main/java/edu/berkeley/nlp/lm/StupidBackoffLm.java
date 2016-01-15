package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.values.CountValueContainer;

/**
 * Language model implementation which uses stupid backoff (Brants et al., 2007)
 * computation. Note that stupid backoff does not properly normalize, so the
 * scores this LM computes are not in fact probabilities. Also, unliked LMs estimated
 * using {@link LmReaders.createKneserNeyLmFromTextFiles}, this model returns natural 
 * logarithms instead of log10.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class StupidBackoffLm<W> extends AbstractArrayEncodedNgramLanguageModel<W> implements ArrayEncodedNgramLanguageModel<W>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected final NgramMap<LongRef> map;

	private final float alpha;

	public StupidBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final NgramMap<LongRef> map, final ConfigOptions opts) {
		super(lmOrder, wordIndexer, (float) opts.unknownWordLogProb);
		this.map = map;
		this.alpha = (float) opts.stupidBackoffAlpha;

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
		final NgramMap<LongRef> localMap = map;
		float logProb = oovWordLogProb;
		long probContext = 0L;
		int probContextOrder = -1;
		long backoffContext = 0L;
		int backoffContextOrder = -1;

		final LongRef scratch = new LongRef(-1L);
		for (int i = endPos - 1; i >= startPos; --i) {
			assert (probContext >= 0);
			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);

			if (probContext < 0) {
				return logProb;
			} else {
				final long currCount = scratch.value;
				long backoffCount = -1L;
				if (i == endPos - 1) {
					backoffCount = ((CountValueContainer) map.getValues()).getUnigramSum();
				} else {
					backoffContext = localMap.getValueAndOffset(backoffContext, backoffContextOrder++, ngram[i], scratch);
					backoffCount = scratch.value;
				}
				logProb = (float) Math.log(currCount / ((float) backoffCount) * pow(alpha, i - startPos));
				probContextOrder++;
			}

		}
		return logProb;
	}

	/**
	 * Gets the raw count of an n-gram.
	 * 
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @return count of n-gram, or -1 if n-gram is not in the map.
	 */
	public long getRawCount(final int[] ngram, final int startPos, final int endPos) {
		final NgramMap<LongRef> localMap = map;
		long probContext = 0L;

		final LongRef scratch = new LongRef(-1L);
		for (int probContextOrder = -1; probContextOrder < endPos - startPos - 1; ++probContextOrder) {
			assert (probContext >= 0);
			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[endPos - probContextOrder - 2], scratch);
			if (probContext < 0) { return -1; }
		}
		return scratch.value;
	}

	private static float pow(final float alpha, final int n) {
		float ret = 1.0f;
		for (int i = 0; i < n; ++i)
			ret *= alpha;
		return ret;
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

	public NgramMap<LongRef> getNgramMap() {
		return map;
	}

}
