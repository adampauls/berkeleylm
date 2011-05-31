package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

public interface CompressibleValueContainer<V> extends ValueContainer<V>
{
	/**
	 * Swaps values at offsets a and b.
	 * 
	 * @param a
	 * @param b
	 * @param ngramOrder
	 */
	public void swap(long a, long b, int ngramOrder);

	/**
	 * Compresses the value at the given offset into a list of bits.
	 * 
	 * @param offset
	 * @param ngramOrder
	 * @return
	 */
	public BitList getCompressed(long offset, int ngramOrder);

	/**
	 * Reads and decompresses from the bit stream bits.
	 * 
	 * @param bits
	 * @param ngramOrder
	 * @param justConsume
	 *            If true, nothing is returned, and the function simply consumes
	 *            the appropriate number of bits from the BitStream.
	 * 
	 * @return
	 */
	public void decompress(BitStream bits, int ngramOrder, boolean justConsume, @OutputParameter V outputVal);

	public void clearStorageAfterCompression(int ngramOrder);

}
