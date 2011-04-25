package edu.berkeley.nlp.lm.map;

import edu.berkeley.nlp.lm.map.AbstractNgramMap.ValueOffsetPair;

public interface OffsetNgramMap<T> extends NgramMap<T>
{
	public long getOffset(int[] ngram, int startPos, int endPos);

	public ValueOffsetPair<T> getValueAndOffset(long contextOffset, int prefixNgramOrder, int word);

}
