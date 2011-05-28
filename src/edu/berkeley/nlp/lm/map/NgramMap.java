package edu.berkeley.nlp.lm.map;

import java.util.List;

import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ValueContainer;

public interface NgramMap<T>
{

	public long put(int[] ngram, int startPos, int endPos, T val);

	public void handleNgramsFinished(int justFinishedOrder);

	public void trim();

	public void initWithLengths(List<Long> numNGrams);

	public ValueContainer<T> getValues();

	public long getValueAndOffset(final long contextOffset, final int contextOrder, int word, @OutputParameter T currProbVal);

}
