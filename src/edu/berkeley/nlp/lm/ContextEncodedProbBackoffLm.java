package edu.berkeley.nlp.lm;

import java.io.Serializable;

import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

/**
 * Language model implementation which uses Kneser-Ney style backoff
 * computation.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class ContextEncodedProbBackoffLm<W> extends AbstractContextEncodedNgramLanguageModel<W> implements ContextEncodedNgramLanguageModel<W>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final HashNgramMap<ProbBackoffPair> map;

	private final ProbBackoffValueContainer values;

	private final long numWords;

	public ContextEncodedProbBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final ContextEncodedNgramMap<ProbBackoffPair> map,
		final ConfigOptions opts) {
		super(lmOrder, wordIndexer, (float) opts.unknownWordLogProb);
		this.map = (HashNgramMap<ProbBackoffPair>) map;
		this.values = (ProbBackoffValueContainer) map.getValues();
		numWords = map.getNumNgrams(0);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.berkeley.nlp.lm.AbstractContextEncodedNgramLanguageModel#getLogProb
	 * (long, int, int,
	 * edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo)
	 */
	@Override
	public float getLogProb(final long contextOffset, final int contextOrder, final int word, @OutputParameter final LmContextInfo outputContext) {
		final HashNgramMap<ProbBackoffPair> localMap = map;
		int currContextOrder = contextOrder;
		long currContextOffset = contextOffset;
		float backoffSum = 0.0f;
		if (word < 0 || word >= numWords) { return oovReturn(outputContext); }
		final boolean onlyUnigram = !localMap.wordHasBigrams(word);
		long longestOffset = -2;
		int longestOrder = -2;

		while (currContextOrder >= -1) {
			final long offset = (onlyUnigram && currContextOrder >= 0) ? -1 : localMap.getOffset(currContextOffset, currContextOrder, word);
			final int ngramOrder = currContextOrder + 1;
			final float prob = offset < 0 ? Float.NaN : values.getProb(ngramOrder, offset);
			if (offset >= 0 && longestOffset == -2) {
				longestOffset = offset;
				longestOrder = ngramOrder;
			}
			if (offset >= 0 && !Float.isNaN(prob)) {
				setOutputContext(outputContext, longestOffset, longestOrder);
				System.out.printf("myprob: %f %f %d %d\n", backoffSum, prob, (int) offset, ngramOrder);
				return backoffSum + prob;
			} else if (currContextOrder >= 0) {
				final long backoffIndex = currContextOffset;
				final float backOff = backoffIndex < 0 ? 0.0f : values.getBackoff(currContextOrder, backoffIndex);
				backoffSum += (Float.isNaN(backOff) ? 0.0f : backOff);
				currContextOffset = currContextOrder == 0 ? 0 : values.getSuffixOffset(currContextOffset, currContextOrder);
			}
			currContextOrder--;
		}
		return oovReturn(outputContext);

	}

	/**
	 * @param outputContext
	 * @return
	 */
	private float oovReturn(final LmContextInfo outputContext) {
		if (outputContext != null) {
			outputContext.offset = 0;
			outputContext.order = -1;
		}
		return oovWordLogProb;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.AbstractContextEncodedNgramLanguageModel#
	 * getOffsetForNgram(int[], int, int)
	 */
	@Override
	public LmContextInfo getOffsetForNgram(final int[] ngram, final int startPos, final int endPos) {
		return map.getOffsetForNgram(ngram, startPos, endPos);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.AbstractContextEncodedNgramLanguageModel#
	 * getNgramForOffset(long, int, int)
	 */
	@Override
	public int[] getNgramForOffset(final long contextOffset, final int contextOrder, final int word) {
		return map.getNgramFromContextEncoding(contextOffset, contextOrder, word);
	}

	public NgramMap<ProbBackoffPair> getNgramMap() {
		return map;
	}

	private void setOutputContext(final LmContextInfo outputContext, final long offset, final int ngramOrder) {
		if (outputContext != null) {
			if (ngramOrder == lmOrder - 1) {
				final long suffixOffset = values.getSuffixOffset(offset, ngramOrder);
				outputContext.offset = suffixOffset;
				outputContext.order = ngramOrder - 1;
			} else {
				outputContext.offset = offset;
				outputContext.order = ngramOrder;

			}
		}
	}

}
