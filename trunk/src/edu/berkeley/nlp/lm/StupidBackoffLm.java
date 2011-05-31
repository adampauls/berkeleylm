package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.values.RankedCountValueContainer;

/**
 * Language model implementation which uses stupid backoff (Brants et al., 2007)
 * computation.
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
		final NgramMap<LongRef> localMap = map;
		float logProb = oovWordLogProb;
		long probContext = 0L;
		int probContextOrder = -1;

		long lastCount = ((RankedCountValueContainer) map.getValues()).getUnigramSum();
		final LongRef scratch = new LongRef(-1L);
		for (int i = endPos_ - 1; i >= startPos_; --i) {
			assert (probContext >= 0);
			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);

			if (probContext < 0) {
				return logProb;
			} else {
				logProb = (float) Math.log(scratch.value / ((float) lastCount) * pow(alpha, i - startPos_));
				lastCount = scratch.value;
				probContextOrder++;
			}

		}
		return logProb;
	}

	private static float pow(final float alpha, final int n) {
		float ret = 1.0f;
		for (int i = 0; i < n; ++i)
			ret *= alpha;
		return ret;
	}

	@Override
	public float getLogProb(final int[] ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	@Override
	public float getLogProb(final List<W> ngram) {
		return ArrayEncodedNgramLanguageModel.DefaultImplementations.getLogProb(ngram, this);
	}

	public NgramMap<LongRef> getNgramMap() {
		return map;
	}

}
