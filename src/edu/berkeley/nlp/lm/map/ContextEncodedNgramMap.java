package edu.berkeley.nlp.lm.map;

import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

public interface ContextEncodedNgramMap<T> extends OffsetNgramMap<T>
{
	public long getOffset(final long contextOffset, final int contextOrder, final int word);

	public long getValueAndOffset(final long contextOffset, final int contextOrder, int word, @OutputParameter T currProbVal);

}
