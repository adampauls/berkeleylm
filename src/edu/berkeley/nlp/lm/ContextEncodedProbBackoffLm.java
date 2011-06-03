package edu.berkeley.nlp.lm;

import java.io.Serializable;

import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
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

	private final ContextEncodedNgramMap<ProbBackoffPair> map;

	private final ProbBackoffValueContainer values;

	public ContextEncodedProbBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final ContextEncodedNgramMap<ProbBackoffPair> map,
		final ConfigOptions opts) {
		super(lmOrder, wordIndexer, (float) opts.unknownWordLogProb);
		this.map = map;
		this.values = (ProbBackoffValueContainer) map.getValues();

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
		final ContextEncodedNgramMap<ProbBackoffPair> localMap = map;
		int currContextOrder = contextOrder;
		long currContextOffset = contextOffset;
		float backoffSum = 0.0f;
		while (true) {
			final long offset = localMap.getOffset(currContextOffset, currContextOrder, word);
			final int ngramOrder = currContextOrder + 1;
			final float prob = offset < 0 ? Float.NaN : values.getProb(ngramOrder, offset);
			if (offset >= 0 && !Float.isNaN(prob)) {
				setOutputContext(outputContext, offset, ngramOrder);
				return backoffSum + prob;
			} else if (currContextOrder >= 0) {
				final long backoffIndex = currContextOffset;
				final float backOff = backoffIndex < 0 ? 0.0f : values.getBackoff(currContextOrder, backoffIndex);
				backoffSum += (Float.isNaN(backOff) ? 0.0f : backOff);
				currContextOrder--;
				currContextOffset = currContextOrder < 0 ? 0 : values.getSuffixOffset(currContextOffset, currContextOrder + 1);
			} else {
				if (outputContext != null) {
					outputContext.offset = 0;
					outputContext.order = -1;
				}
				return oovWordLogProb;
			}
		}
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
