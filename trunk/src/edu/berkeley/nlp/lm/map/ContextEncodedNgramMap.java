package edu.berkeley.nlp.lm.map;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

public interface ContextEncodedNgramMap<T> extends NgramMap<T>
{
	public long getOffset(final long contextOffset, final int contextOrder, final int word);

	public long getOffset(int[] ngram, int startPos, int endPos);

	public LmContextInfo getOffsetForNgram(int[] ngram, int startPos, int endPos);

}
