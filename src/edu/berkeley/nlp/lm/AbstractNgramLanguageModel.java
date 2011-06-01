package edu.berkeley.nlp.lm;

import java.util.List;

public abstract class AbstractNgramLanguageModel<W> implements NgramLanguageModel<W>
{

	private static final long serialVersionUID = 1L;

	protected final int lmOrder;

	private final WordIndexer<W> wordIndexer;

	/**
	 * Fixed constant returned when computing the log probability for an n-gram
	 * whose last word is not in the vocabulary. Note that this is different
	 * from the log prob of the <code>unk</code> tag probability.
	 * 
	 */
	protected final float oovWordLogProb;

	public AbstractNgramLanguageModel(final int lmOrder, final WordIndexer<W> wordIndexer, final float oovWordLogProb) {
		this.lmOrder = lmOrder;
		this.wordIndexer = wordIndexer;
		this.oovWordLogProb = oovWordLogProb;
	}

	@Override
	public int getLmOrder() {
		return lmOrder;
	}

	@Override
	public WordIndexer<W> getWordIndexer() {
		return wordIndexer;
	}

}
