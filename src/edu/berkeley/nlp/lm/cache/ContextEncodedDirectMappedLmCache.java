package edu.berkeley.nlp.lm.cache;

import java.util.Arrays;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Logger;

public final class ContextEncodedDirectMappedLmCache implements ContextEncodedLmCache
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static final int VAL_OFFSET = 0;
	
	private static final int WORD_OFFSET = 1;

	private static final int CONTEXT_OFFSET = 2;

	private static final int CONTEXT_ORDER = 4;

	private static final int OUTPUT_CONTEXT_OFFSET = 5;

	private static final int OUTPUT_CONTEXT_ORDER = 7;


	private static final int STRUCT_LENGTH = 8;


	// for efficiency, this array fakes a struct with fields
	// float prob;
	// int word;
	// long contextOffset;
	// int contextOrder;
	// long outputContextOffset;
	// int outputContextOrder;
	private final int[] array;

	private final int cacheSize;

	public ContextEncodedDirectMappedLmCache(final int cacheBits) {
		cacheSize = (1 << cacheBits) - 1;
		array = new int[STRUCT_LENGTH * cacheSize];
		Arrays.fill(array, Float.floatToIntBits(Float.NaN));
	}

	@Override
	public float getCached(final long contextOffset, final int contextOrder, final int word, final int hash, @OutputParameter final LmContextInfo outputPrefix) {
		final float f = getVal(hash);
		final long outputContextOffset = getOutputContextOffset(hash);
		if (!Float.isNaN(f) && (outputPrefix == null || outputContextOffset >= 0)) {
			final int cachedWordHere = getWord(hash);
			if (cachedWordHere != -1 && equals(contextOffset, contextOrder, word, getContextOffset(hash), cachedWordHere, getContextOrder(hash))) {
				if (outputPrefix != null) {
					outputPrefix.order = getOutputContextOrder(hash);
					outputPrefix.offset = outputContextOffset;
				}
				return f;
			}
		}
		return Float.NaN;
	}

	private boolean equals(final long contextOffset, final int contextOrder, final int word, final long cachedOffsetHere, final int cachedWordHere,
		final int cachedOrderHere) {
		return word == cachedWordHere && contextOrder == cachedOrderHere && contextOffset == cachedOffsetHere;
	}

	@Override
	public void putCached(final long contextOffset, final int contextOrder, final int word, final float score, final int hash,
		@OutputParameter final LmContextInfo outputPrefix) {

		setWord(hash, word);
		setVal(hash, score);
		setContextOffset(hash, contextOffset);
		setContextOrder(hash, contextOrder);
		setOutputContextOrder(hash, outputPrefix == null ? -1 : outputPrefix.order);
		setOutputContextOffset(hash, outputPrefix == null ? -1 : outputPrefix.offset);

	}

	private int getWord(int hash) {
		return array[startOfStruct(hash) + WORD_OFFSET];
	}

	private int getContextOrder(int hash) {
		return array[startOfStruct(hash) + CONTEXT_ORDER];
	}

	private int getOutputContextOrder(int hash) {
		return array[startOfStruct(hash) + OUTPUT_CONTEXT_ORDER];
	}

	private long getOutputContextOffset(int hash) {
		return getLong(hash, OUTPUT_CONTEXT_OFFSET);
	}

	private long getContextOffset(int hash) {
		return getLong(hash, CONTEXT_OFFSET);
	}

	/**
	 * @param hash
	 * @param off
	 * @return
	 */
	private long getLong(int hash, final int off) {
		return (((long) array[startOfStruct(hash) + off + 1]) << Integer.SIZE) | array[startOfStruct(hash) + off];
	}

	private float getVal(int hash) {
		return Float.intBitsToFloat(array[startOfStruct(hash) + VAL_OFFSET]);
	}

	private void setWord(int hash, int word) {
		array[startOfStruct(hash) + WORD_OFFSET] = word;
	}

	private void setContextOrder(int hash, int order) {
		array[startOfStruct(hash) + CONTEXT_ORDER] = order;
	}

	private void setOutputContextOrder(int hash, int order) {
		array[startOfStruct(hash) + OUTPUT_CONTEXT_ORDER] = order;
	}

	private void setOutputContextOffset(int hash, long offset) {
		final int off = OUTPUT_CONTEXT_OFFSET;
		setLong(hash, offset, off);
	}

	/**
	 * @param hash
	 * @param l
	 * @param off
	 */
	private void setLong(int hash, long l, final int off) {
		array[startOfStruct(hash) + off] = (int) (l);
		array[startOfStruct(hash) + off + 1] = (int) (l >>> Integer.SIZE);
	}

	private void setContextOffset(int hash, long offset) {
		final int off = CONTEXT_OFFSET;
		setLong(hash, offset, off);
	}

	private void setVal(int hash, float f) {
		array[startOfStruct(hash) + VAL_OFFSET] = Float.floatToIntBits(f);
	}

	private static int startOfStruct(int hash) {
		return hash * STRUCT_LENGTH;
	}

	@Override
	public int capacity() {
		return cacheSize;
	}
}
