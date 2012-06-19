package edu.berkeley.nlp.lm;

import java.io.Serializable;

import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

public abstract class AbstractNgramLanguageModel<W> implements NgramLanguageModel<W>, Serializable

{

	

	private static final long serialVersionUID = 1L;

	protected final int lmOrder;

	@PrintMemoryCount
	private final WordIndexer<W> wordIndexer;

	/**
	 * Fixed constant returned when computing the log probability for an n-gram
	 * whose last word is not in the vocabulary. Note that this is different
	 * from the log prob of the <code>unk</code> tag probability.
	 * 
	 */
	protected float oovWordLogProb;

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
	
	@Override
	public void setOovWordLogProb(float oovWordLogProb) {
		this.oovWordLogProb = oovWordLogProb;
	}

}
