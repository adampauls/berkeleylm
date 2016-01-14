package edu.berkeley.nlp.lm.cache;

import java.io.Serializable;

public interface ArrayEncodedLmCache extends Serializable
{

	/**
	 * Should return Float.NaN if the requested n-gram is not in the cache
	 * 
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param shortHash
	 * @return
	 */
	public float getCached(int[] ngram, int startPos, int endPos, int hash);

	public void clear();

	public void putCached(int[] ngram, int startPos, int endPos, float f, int hash);

	/**
	 * How n-grams can be cached (at most).
	 * 
	 * @return
	 */
	public int capacity();

}