package edu.berkeley.nlp.lm.cache;

import java.util.Arrays;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

public final class ContextEncodedDirectMappedLmCache implements ContextEncodedLmCache
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static final int VAL_OFFSET = 0;

	private static final int OUTPUT_CONTEXT_OFFSET = 1;

	private static final int WORD_OFFSET = 3;

	private static final int CONTEXT_OFFSET = 4;

	private static final int CONTEXT_ORDER = 6;

	private static final int OUTPUT_CONTEXT_ORDER = 7;

	private static final int STRUCT_LENGTH = 8;

	// for efficiency, this array fakes a struct with fields
	// float prob;
	// long outputContextOffset;
	// int outputContextOrder;
	// int word;
	// long contextOffset;
	// int contextOrder;
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

		if (!Float.isNaN(f)) {
			long outputContextOffset = -1;
			if (outputPrefix == null || (outputContextOffset = getOutputContextOffset(hash)) >= 0) {
				final int cachedWordHere = getWord(hash);

				if (cachedWordHere != -1 && word == cachedWordHere) {
					if (contextOffset == getContextOffset(hash) & contextOrder == getContextOrder(hash)) {
						if (outputPrefix != null) {
							outputPrefix.order = getOutputContextOrder(hash);
							outputPrefix.offset = outputContextOffset;
						}
						return f;
					}
				}
			}
		}
		return Float.NaN;
	}

	@Override
	public void putCached(final long contextOffset, final int contextOrder, final int word, final float score, final int hash,
		@OutputParameter final LmContextInfo outputPrefix) {

		setVal(hash, score);
		setOutputContextOffset(hash, outputPrefix == null ? -1 : outputPrefix.offset);
		setWord(hash, word);
		setContextOffset(hash, contextOffset);
		setContextOrder(hash, contextOrder);
		setOutputContextOrder(hash, outputPrefix == null ? -1 : outputPrefix.order);

	}

	private int getWord(final int hash) {
		return array[startOfStruct(hash) + WORD_OFFSET];
	}

	private int getContextOrder(final int hash) {
		return array[startOfStruct(hash) + CONTEXT_ORDER];
	}

	private int getOutputContextOrder(final int hash) {
		return array[startOfStruct(hash) + OUTPUT_CONTEXT_ORDER];
	}

	private long getOutputContextOffset(final int hash) {
		return getLong(hash, OUTPUT_CONTEXT_OFFSET);
	}

	private long getContextOffset(final int hash) {
		return getLong(hash, CONTEXT_OFFSET);
	}

	/**
	 * @param hash
	 * @param off
	 * @return
	 */
	private long getLong(final int hash, final int off) {
		return (((long) array[startOfStruct(hash) + off + 1]) << Integer.SIZE) | (((1L << Integer.SIZE) - 1) & array[startOfStruct(hash) + off]);
	}

	private float getVal(final int hash) {
		return Float.intBitsToFloat(array[startOfStruct(hash) + VAL_OFFSET]);
	}

	private void setWord(final int hash, final int word) {
		array[startOfStruct(hash) + WORD_OFFSET] = word;
	}

	private void setContextOrder(final int hash, final int order) {
		array[startOfStruct(hash) + CONTEXT_ORDER] = order;
	}

	private void setOutputContextOrder(final int hash, final int order) {
		array[startOfStruct(hash) + OUTPUT_CONTEXT_ORDER] = order;
	}

	private void setOutputContextOffset(final int hash, final long offset) {
		final int off = OUTPUT_CONTEXT_OFFSET;
		setLong(hash, offset, off);
	}

	/**
	 * @param hash
	 * @param l
	 * @param off
	 */
	private void setLong(final int hash, final long l, final int off) {
		array[startOfStruct(hash) + off] = (int) (l);
		array[startOfStruct(hash) + off + 1] = (int) (l >>> Integer.SIZE);
	}

	private void setContextOffset(final int hash, final long offset) {
		final int off = CONTEXT_OFFSET;
		setLong(hash, offset, off);
	}

	private void setVal(final int hash, final float f) {
		array[startOfStruct(hash) + VAL_OFFSET] = Float.floatToIntBits(f);
	}

	private static int startOfStruct(final int hash) {
		return hash * STRUCT_LENGTH;
	}

	@Override
	public int capacity() {
		return cacheSize;
	}
}
