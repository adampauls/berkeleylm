package edu.berkeley.nlp.lm.map;

import edu.berkeley.nlp.lm.map.AbstractNgramMap.ValueOffsetPair;

public interface ContextEncodedNgramMap<T> extends OffsetNgramMap<T>
{
	public long getOffset(final long contextOffset, final int contextOrder, final int word);

}
