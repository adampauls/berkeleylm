package edu.berkeley.nlp.lm;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

/**
 * 
 * Default implementation of all ContextEncodedNgramLanguageModel functionality
 * except {@link #getLogProb(long, int, int, LmContextInfo)},
 * {@link #getOffsetForNgram(int[], int, int), and {
 * @link #getNgramForOffset(long, int, int)}.
 * 
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public abstract class AbstractContextEncodedNgramLanguageModel<W> extends AbstractNgramLanguageModel<W> implements ContextEncodedNgramLanguageModel<W>,
	Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public AbstractContextEncodedNgramLanguageModel(final int lmOrder, final WordIndexer<W> wordIndexer, final float oovWordLogProb) {
		super(lmOrder, wordIndexer, oovWordLogProb);
	}

	@Override
	public float scoreSentence(final List<W> sentence) {
		return ContextEncodedNgramLanguageModel.DefaultImplementations.scoreSentence(sentence, this);
	}

	@Override
	public float getLogProb(final List<W> phrase) {
		return ContextEncodedNgramLanguageModel.DefaultImplementations.getLogProb(phrase, this);
	}

	@Override
	public abstract float getLogProb(long contextOffset, int contextOrder, int word, @OutputParameter LmContextInfo outputContext);

	@Override
	public abstract LmContextInfo getOffsetForNgram(int[] ngram, int startPos, int endPos);

	@Override
	public abstract int[] getNgramForOffset(long contextOffset, int contextOrder, int word);

}
