package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.ConfigOptions;
import edu.berkeley.nlp.lm.map.OffsetNgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.values.CountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

/**
 * Language model implementation which uses Kneser-Ney-style backoff
 * computation.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class StupidBackoffLm<W> extends AbstractNgramLanguageModel<W> implements NgramLanguageModel<W>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected final NgramMap<LongRef> map;

	/**
	 * Fixed constant returned when computing the log probability for an n-gram
	 * whose last word is not in the vocabulary. Note that this is different
	 * from the log prob of the <code>unk</code> tag probability.
	 * 
	 */
	private final float oovWordLogProb;

	private final float alpha;

	public StupidBackoffLm(final int lmOrder, final WordIndexer<W> wordIndexer, final NgramMap<LongRef> map, final ConfigOptions opts) {
		super(lmOrder, wordIndexer);
		oovWordLogProb = (float) opts.unknownWordLogProb;
		this.map = map;
		this.alpha = (float) opts.stupidBackoffAlpha;

	}

	@Override
	public float getLogProb(final int[] ngram, final int startPos_, final int endPos_) {
		//		if (map instanceof OffsetNgramMap<?>) {
		//			return getLogProbWithOffsets(ngram, startPos_, endPos_);
		//		} else {
		return getLogProbDirectly(ngram, startPos_, endPos_);
		//		}
	}

	/**
	 * @param ngram
	 * @param startPos_
	 * @param endPos_
	 * @return
	 */
	private float getLogProbDirectly(final int[] ngram, final int startPos, final int endPos) {
		final ContextEncodedNgramMap<LongRef> localMap = (ContextEncodedNgramMap<LongRef>) map;
		float logProb = oovWordLogProb;
		long probContext = 0L;
		int probContextOrder = -1;

		long lastCount = ((CountValueContainer) map.getValues()).getUnigramSum();
		LongRef scratch = new LongRef(-1L);
		for (int i = endPos - 1; i >= startPos; --i) {
			assert (probContext >= 0);
			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);

			if (probContext < 0) {
				return logProb;
			} else {
				logProb = scratch.value / lastCount * pow(alpha, i - startPos);
				lastCount = scratch.value;
				probContextOrder++;

			}
			if (i == startPos) break;

		}
		return logProb;
	}

	//	/**
	//	 * @param ngram
	//	 * @param startPos_
	//	 * @param endPos_
	//	 * @return
	//	 */
	//	private float getLogProbWithOffsets(final int[] ngram, final int startPos, final int endPos) {
	//		float logProb = oovWordLogProb;
	//		float backoff = 0.0f;
	//
	//		ContextEncodedNgramMap<ProbBackoffPair> localMap = (ContextEncodedNgramMap<ProbBackoffPair>) map;
	//		long probContext = 0L;
	//		int probContextOrder = -1;
	//
	//		long backoffContext = 0L;
	//		int backoffContextOrder = -1;
	//
	//		for (int i = endPos - 1; i >= startPos; --i) {
	//			if (probContext >= 0) probContext = localMap.getOffset(probContext, probContextOrder, ngram[i]);
	//			if (probContext >= 0) {
	//				probContextOrder++;
	//				final float currProb = values.getProb(probContextOrder, probContext);
	//				if (Float.isNaN(currProb) && i == startPos) {
	//					return logProb + backoff;
	//				} else if (!Float.isNaN(currProb)) {
	//
	//					logProb = currProb;
	//					backoff = 0.0f;
	//				}
	//			}
	//			if (i == startPos) break;
	//
	//			backoffContext = localMap.getOffset(backoffContext, backoffContextOrder, ngram[i - 1]);
	//			if (backoffContext < 0) break;
	//
	//			backoffContextOrder++;
	//			final float currBackoff = values.getBackoff(backoffContextOrder, backoffContext);
	//			backoff += Float.isNaN(currBackoff) ? 0.0f : currBackoff;
	//		}
	//		return logProb + backoff;
	//	}

	private static float pow(float alpha, int n) {
		float ret = 1.0f;
		for (int i = 0; i < n; ++i)
			ret *= alpha;
		return ret;
	}

	@Override
	public WordIndexer<W> getWordIndexer() {
		return wordIndexer;
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

}
