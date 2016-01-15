package edu.berkeley.nlp.lm.io;

import java.util.List;

/**
 * Callback that is called for each n-gram in the collection
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type for each n-gram (either count of prob/backoff)
 */
public interface ArpaLmReaderCallback<V> extends NgramOrderedLmReaderCallback<V>
{

	/**
	 * Called initially with a list of how many n-grams will appear for each
	 * order.
	 * 
	 * @param numNGrams
	 *            maps n-gram orders to number of n-grams (i.e. numNGrams.get(0)
	 *            is the number of unigrams)
	 */
	public void initWithLengths(List<Long> numNGrams);
}