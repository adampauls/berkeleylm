package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.NgramMapOpts;
import edu.berkeley.nlp.lm.map.OffsetNgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

/**
 * Language model implementation which uses Katz-style backoff computation.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class KatzBackoffLm<W> extends AbstractContextEncodedNgramLanguageModel<W> implements NgramLanguageModel<W>, ContextEncodedNgramLanguageModel<W>,
	Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected final NgramMap<ProbBackoffPair> map;

	private final ProbBackoffValueContainer values;

	/**
	 * Fixed constant returned when computing the log probability for an n-gram
	 * whose last word is not in the vocabulary. Note that this is different
	 * from the log prob of the <code>unk</code> tag probability.
	 * 
	 */
	private final float oovWordLogProb;

	public KatzBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final NgramMap<ProbBackoffPair> map, final NgramMapOpts opts) {
		super(lmOrder, wordIndexer);
		oovWordLogProb = (float) opts.unknownWordLogProb;
		this.map = map;
		this.values = (ProbBackoffValueContainer) map.getValues();

	}

	@Override
	public float getLogProb(final int[] ngram, final int startPos, final int endPos) {
		//		if (map.isReversed()) { return scoreLog10HelpReversed(ngram, startPos, endPos); }
		final float score = scoreLog10Help(ngram, startPos, endPos);
		return score;
	}

	@Override
	public float getLogProb(final long context, final int contextOrder, final int word, final LmContextInfo outputPrefixIndex) {
		final float score = getLogProbContextEncoded(context, contextOrder, word, outputPrefixIndex);
		final float log10ln = score;//convertLog ? convertFromLogBase10(score) : score;
		return log10ln;
	}

	private float scoreLog10Help(final int[] ngram, final int startPos, final int endPos) {

		if (map instanceof OffsetNgramMap<?>) {

			final OffsetNgramMap<ProbBackoffPair> localMap = (OffsetNgramMap<ProbBackoffPair>) map;

			final long index = localMap.getOffset(ngram, startPos, endPos);
			if (index >= 0) {
				final int ngramOrder = endPos - startPos - 1;
				final float prob = values.getProb(ngramOrder, index);
				if (!Float.isNaN(prob)) {

				return prob; }
			}
			if (endPos - startPos > 1) {
				final float backoffProb = getLogProb(ngram, startPos + 1, endPos);
				final long backoffIndex = localMap.getOffset(ngram, startPos, endPos - 1);
				float backOff = backoffIndex < 0 ? 0.0f : values.getBackoff(endPos - startPos - 2, backoffIndex);
				backOff = Float.isNaN(backOff) ? 0.0f : backOff;
				return backOff + backoffProb;
			} else {
				return oovWordLogProb;
			}

		} else {
			final ProbBackoffPair pair = map.getValue(ngram, startPos, endPos, null);
			if (pair != null && !Float.isNaN(pair.prob)) {
				return pair.prob;
			} else {
				if (endPos - startPos > 1) {
					final float backoffProb = getLogProb(ngram, startPos + 1, endPos);
					final ProbBackoffPair backoffPair = map.getValue(ngram, startPos, endPos - 1, null);
					final float backOff = backoffPair == null ? 0.0f : backoffPair.backoff;
					return backOff + backoffProb;
				} else {
					return oovWordLogProb;
				}
			}

		}
	}

	private float scoreLog10HelpReversed(final int[] ngram, final int startPos, final int endPos) {

		if (map instanceof ContextEncodedNgramMap<?>) {
			final ContextEncodedNgramMap<ProbBackoffPair> localMap = (ContextEncodedNgramMap<ProbBackoffPair>) map;
			float logProb = oovWordLogProb;
			float backoff = 0.0f;

			long probContext = 0L;
			int probContextOrder = -1;

			long backoffContext = 0L;
			int backoffContextOrder = -1;

			for (int i = endPos - 1; i >= startPos; --i) {

				if (probContext >= 0) probContext = localMap.getOffset(probContext, probContextOrder, ngram[i]);
				if (probContext >= 0) {
					probContextOrder++;
					final float currProb = values.getProb(probContextOrder, probContext);
					if (Float.isNaN(currProb) && i == startPos) {
						return logProb + backoff;
					} else if (!Float.isNaN(currProb)) {

						logProb = currProb;
						backoff = 0.0f;
					}
				}
				if (i == startPos) break;

				backoffContext = localMap.getOffset(backoffContext, backoffContextOrder, ngram[i - 1]);
				if (backoffContext < 0) break;

				backoffContextOrder++;
				final float currBackoff = values.getBackoff(backoffContextOrder, backoffContext);
				backoff += Float.isNaN(currBackoff) ? 0.0f : currBackoff;

			}

			return logProb + backoff;
		} else {
			// TODO implement this
			assert false;
			return Float.NaN;
			//			float logProb = oovWordLogProb;
			//			float backoff = 0.0f;
			//
			//			long probContext = 0L;
			//			int probContextOrder = -1;
			//
			//			long backoffContext = 0L;
			//			int backoffContextOrder = -1;
			//
			//			for (int i = endPos - 1; i >= startPos; --i) {
			//
			//				ProbBackoffPair currProbVal = null;
			//				if (probContext >= 0) {
			//					final ValueOffsetPair<ProbBackoffPair> probValueAndOffset = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i]);
			//					probContext = probValueAndOffset.getOffset();
			//					currProbVal = probValueAndOffset.getValue();
			//				}
			//				if (probContext >= 0) {
			//					assert currProbVal != null;
			//					probContextOrder++;
			//					final float currProb = currProbVal.prob;
			//					if (Float.isNaN(currProb) && i == startPos) {
			//						return logProb + backoff;
			//					} else if (!Float.isNaN(currProb)) {
			//
			//						logProb = currProb;
			//						backoff = 0.0f;
			//					}
			//				}
			//				if (i == startPos) break;
			//
			//				final ValueOffsetPair<ProbBackoffPair> backoffValueAndOffset = localMap.getValueAndOffset(backoffContext, backoffContextOrder, ngram[i - 1]);
			//				backoffContext = backoffValueAndOffset.getOffset();
			//				if (backoffContext < 0) break;
			//
			//				backoffContextOrder++;
			//				final float currBackoff = backoffValueAndOffset.getValue().backoff;
			//				backoff += Float.isNaN(currBackoff) ? 0.0f : currBackoff;
			//
			//			}
			//
			//			return logProb + backoff;

		}

	}

	private float getLogProbContextEncoded(final long contextOffset, final int contextOrder, final int word, @OutputParameter final LmContextInfo outputContext) {

		if (map instanceof ContextEncodedNgramMap<?>) {

			final ContextEncodedNgramMap<ProbBackoffPair> localMap = (ContextEncodedNgramMap<ProbBackoffPair>) map;

			final long index = localMap.getOffset(contextOffset, contextOrder, word);
			if (index >= 0) {
				final int ngramOrder = contextOrder + 1;
				final float prob = values.getProb(ngramOrder, index);
				if (outputContext != null) {
					if (ngramOrder < lmOrder - 1) {
						outputContext.offset = index;
						outputContext.order = ngramOrder;
					} else {
						outputContext.offset = values.getContextOffset(index, ngramOrder);
						outputContext.order = contextOrder;
					}
					assert ngramOrder < lmOrder;
				}
				return prob;
			} else if (contextOrder >= 0) {
				final int nextPrefixOrder = contextOrder - 1;
				final long nextPrefixIndex = nextPrefixOrder < 0 ? 0 : values.getContextOffset(contextOffset, contextOrder);
				final float nextProb = getLogProb(nextPrefixIndex, nextPrefixOrder, word, outputContext);
				final long backoffIndex = contextOffset;
				final float backOff = backoffIndex < 0 ? 0.0f : values.getBackoff(contextOrder, backoffIndex);
				return backOff + nextProb;
			} else {
				return oovWordLogProb;
			}
		} else {
			// TODO set up compressed version for context-encoded querying
			throw new RuntimeException("Compressed version not set up for context-encoded querying yet");
		}
	}

	@Override
	public WordIndexer<W> getWordIndexer() {
		return wordIndexer;
	}

	@Override
	public int[] getNgramForContext(final long contextOffset, final int contextOrder) {
		// TODO Auto-generated method stub
		throw new UnsupportedOperationException("Method not yet implemented");
	}

	@Override
	public float getLogProb(final int[] ngram) {
		return NgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	@Override
	public float getLogProb(final List<W> ngram) {
		return NgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	@Override
	public float scoreSequence(final List<W> sequence) {
		return NgramLanguageModel.DefaultImplementations.scoreSequence(sequence, this);
	}

	@Override
	public LmContextInfo getOffsetForNgram(int[] ngram, int startPos, int endPos) {
		return map.getOffsetForNgram(ngram, startPos,  endPos);
	}

}
