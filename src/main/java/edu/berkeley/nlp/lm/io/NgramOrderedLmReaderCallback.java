package edu.berkeley.nlp.lm.io;

/**
 * Callback that is called for each n-gram in the collection
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type for each n-gram (either count of prob/backoff)
 */
public interface NgramOrderedLmReaderCallback<V> extends LmReaderCallback<V>
{

	/**
	 * Called when all n-grams of a given order are finished
	 * 
	 * @param order
	 */
	public void handleNgramOrderFinished(int order);

	/**
	 * Called when n-grams of a given order are started
	 * 
	 * @param order
	 */
	public void handleNgramOrderStarted(int order);

}