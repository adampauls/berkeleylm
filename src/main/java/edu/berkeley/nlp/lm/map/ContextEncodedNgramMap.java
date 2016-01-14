package edu.berkeley.nlp.lm.map;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;

public interface ContextEncodedNgramMap<T> extends NgramMap<T>
{
	public long getOffset(final long contextOffset, final int contextOrder, final int word);

	public LmContextInfo getOffsetForNgram(int[] ngram, int startPos, int endPos);

	public int[] getNgramFromContextEncoding(long contextOffset, int contextOrder, int word);

	public boolean wordHasBigrams(int word);

}
