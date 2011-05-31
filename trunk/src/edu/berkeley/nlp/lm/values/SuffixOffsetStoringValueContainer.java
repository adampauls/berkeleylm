package edu.berkeley.nlp.lm.values;

public interface SuffixOffsetStoringValueContainer<V> extends ValueContainer<V>
{

	/**
	 * Retrieves a stored suffix offset for a n-gram (given by offset)
	 * 
	 * @param offset
	 * @param ngramOrder
	 * @return
	 */
	public long getContextOffset(long offset, int ngramOrder);

}
