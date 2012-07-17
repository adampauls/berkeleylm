package edu.berkeley.nlp.lm;

import java.io.Serializable;

import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;
import edu.berkeley.nlp.lm.values.UncompressedProbBackoffValueContainer;

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
		if (word < 0 || word >= numWords) { return oovReturn(outputContext); }

		final HashNgramMap<ProbBackoffPair> localMap = map;
		long longestOffset = -2;
		int longestOrder = -2;
		float backoffSum = 0.0f;

		long currContextOffset = contextOffset;
		for (int currContextOrder = contextOrder; currContextOrder >= 0; --currContextOrder) {
			final int ngramOrder = currContextOrder + 1;
			final long offset = localMap.getOffset(currContextOffset, currContextOrder, word);
			if (offset >= 0) {
				if (longestOffset == -2) {
					longestOffset = offset;
					longestOrder = ngramOrder;
				}
				final float prob = values.getProb(ngramOrder, offset);
				if (!Float.isNaN(prob)) {
					setOutputContext(outputContext, longestOffset, longestOrder);
					return backoffSum + prob;
				}
			}
			final float backOff = values.getBackoff(currContextOrder, currContextOffset);
			backoffSum += (Float.isNaN(backOff) ? 0.0f : backOff);
			if (currContextOrder > 0) currContextOffset = values.getSuffixOffset(currContextOffset, currContextOrder);
		}

		// do unigram
		final long offset = word;
		final int ngramOrder = 0;
		final float prob = values.getProb(ngramOrder, offset);
		if (Float.isNaN(prob)) return oovReturn(outputContext);
		setOutputContext(outputContext, longestOffset == -2 ? offset : longestOffset, longestOffset == -2 ? ngramOrder : longestOrder);
		return backoffSum + prob;

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
