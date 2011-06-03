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

	public int getMaxNgramOrder();
	
	public long getNumNgrams(int ngramOrder);

	public Iterable<Entry<T>> getNgramsForOrder(final int ngramOrder);

	public static class Entry<T>
	{
		/**
		 * @param key
		 * @param value
		 */
		public Entry(int[] key, T value) {
			super();
			this.key = key;
			this.value = value;
		}

		public int[] key;

		public T value;

	}

}
