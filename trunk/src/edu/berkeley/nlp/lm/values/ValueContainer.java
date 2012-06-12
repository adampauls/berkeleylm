package edu.berkeley.nlp.lm.values;

import java.io.Serializable;

import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

/**
 * Manages storage of arbitrary values in an NgramMap
 * 
 * @author adampauls
 * 
 * @param <V>
 */
public interface ValueContainer<V> extends Serializable
{

	/**
	 * Adds a new value at the specified offset.
	 * 
	 * @param ngramOrder
	 *            As always, ngramOrder is 0-based (0=unigram)
	 * @param offset
	 * @param contextOffset
	 * @param word
	 * @param val
	 * @param suffixOffset
	 * @return Whether or not the add was successful
	 */
	public boolean add(int[] ngram, int startPos, int endPos, int ngramOrder, long offset, long contextOffset, int word, V val, long suffixOffset,
		boolean ngramIsNew);

	/**
	 * Sets internal storage for size for a particular n-gram order
	 * 
	 * @param size
	 * @param ngramOrder
	 */
	public void setSizeAtLeast(long size, int ngramOrder);

	/**
	 * Creates a fresh value container for copying purposes.
	 * 
	 * @return
	 */
	public ValueContainer<V> createFreshValues(long[] numNgramsForEachOrder);

	/**
	 * Gets the value living at a particular offset.
	 * 
	 * @param offset
	 * @param ngramOrder
	 * @return
	 */
	public void getFromOffset(long offset, int ngramOrder, @OutputParameter V outputVal);

	/**
	 * Destructively sets internal storage from another object.
	 * 
	 * @param other
	 */
	public void setFromOtherValues(ValueContainer<V> other);

	/**
	 * Clear storage after an n-gram order is complete
	 * 
	 * @param ngramOrder
	 * @param size
	 */
	public void trimAfterNgram(int ngramOrder, long size);

	/**
	 * Final clean up of storage.
	 */
	public void trim();

	/**
	 * Creates a fresh value of object (useful for passing as an output
	 * parameter)
	 * 
	 * @return
	 */
	public V getScratchValue();

	/**
	 * Initializes a value container with the map that contains it
	 */
	public void setMap(NgramMap<V> map);

	public void clearStorageForOrder(final int ngramOrder);

	public boolean storeSuffixoffsets();
	
	public int numValueBits(int ngramOrder);

}