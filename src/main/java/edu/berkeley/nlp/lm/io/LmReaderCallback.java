package edu.berkeley.nlp.lm.io;


/**
 * Callback that is called for each n-gram in the collection
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type for each n-gram (either count of prob/backoff)
 */
public interface LmReaderCallback<V>
{

	/**
	 * Called for each n-gram
	 * 
	 * @param ngram
	 *            The integer representation of the words as given by the
	 *            provided WordIndexer
	 * @param value
	 *            The value of the n-gram
	 * @param words
	 *            The string representation of the n-gram (space separated)
	 */
	public void call(int[] ngram, int startPos, int endPos, V value, String words);

	/**
	 * Called once all reading is done.
	 */
	public void cleanup();

}