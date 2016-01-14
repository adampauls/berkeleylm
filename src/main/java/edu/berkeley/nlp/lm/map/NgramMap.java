package edu.berkeley.nlp.lm.map;

import java.util.List;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ValueContainer;

public interface NgramMap<V>
{

	public long put(int[] ngram, int startPos, int endPos, V val);

	public void handleNgramsFinished(int justFinishedOrder);

	public void trim();

	public void initWithLengths(List<Long> numNGrams);

	public ValueContainer<V> getValues();

	public long getValueAndOffset(final long contextOffset, final int contextOrder, int word, @OutputParameter V currProbVal);

	public int getMaxNgramOrder();

	public long getNumNgrams(int ngramOrder);

	public Iterable<Entry<V>> getNgramsForOrder(final int ngramOrder);

	public CustomWidthArray getValueStoringArray(final int ngramOrder);

	public static class Entry<T>
	{
		/**
		 * @param key
		 * @param value
		 */
		public Entry(final int[] key, final T value) {
			super();
			this.key = key;
			this.value = value;
		}

		public int[] key;

		public T value;

	}

	public boolean contains(int[] ngram, int startPos, int endPos);

	public V get(int[] ngram, int startPos, int endPos);

	public void clearStorage();

}
