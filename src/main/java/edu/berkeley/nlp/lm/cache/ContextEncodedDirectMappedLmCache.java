package edu.berkeley.nlp.lm.cache;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicIntegerArray;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

public final class ContextEncodedDirectMappedLmCache implements ContextEncodedLmCache
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static int pos = 0;

	private static final int VAL_AND_WORD_OFFSET = pos++;

	private static final int CONTEXT_OFFSET = pos++;

	private static final int OUTPUT_CONTEXT_OFFSET = pos++;

	private static final int STRUCT_LENGTH = pos;

	private static int NUM_ORDER_BITS = 4;// good for up to 16-grams

	private static int NUM_OFFSETS_BITS = (Long.SIZE - NUM_ORDER_BITS);

	private static long ORDER_BIT_MASK = ((1L << NUM_ORDER_BITS) - 1) << (NUM_OFFSETS_BITS);

	private static long OFFSET_BIT_MASK = ((1L << NUM_OFFSETS_BITS) - 1);

	private static long WORD_MASK = ((1L << Integer.SIZE) - 1) << Integer.SIZE;

	private static long FLOAT_MASK = ((1L << Integer.SIZE) - 1);

	// for efficiency, this array fakes a struct with fields
	// float prob;
	// int word;
	// long contextOffset; (also contains order of context)
	// long outputContextOffset; (also contains order of context)
	private final long[] threadUnsafeArray;

	private final ThreadLocal<long[]> threadSafeArray;

	private final int cacheSize;

	private final boolean threadSafe;

	public ContextEncodedDirectMappedLmCache(final int cacheBits, final boolean threadSafe) {
		cacheSize = (1 << cacheBits) - 1;
		this.threadSafe = threadSafe;
		if (threadSafe) {
			threadUnsafeArray = null;
			threadSafeArray = new ThreadLocal<long[]>()
			{
				@Override
				protected long[] initialValue() {
					return allocCache();
				}

			};
		} else {
			threadSafeArray = null;
			threadUnsafeArray = allocCache();
		}

	}

	/**
	 * @return
	 */
	private long[] allocCache() {
		final long[] array = new long[STRUCT_LENGTH * cacheSize];
		Arrays.fill(array, -1);
		return array;
	}

	@Override
	public float getCached(final long contextOffset, final int contextOrder, final int word, final int hash, @OutputParameter final LmContextInfo outputPrefix) {
		final long[] array = !threadSafe ? threadUnsafeArray : threadSafeArray.get();
		final int cachedWordHere = getWord(hash, array);

		if (word >= 0 && word == cachedWordHere && getLong(hash, CONTEXT_OFFSET, array) == combine(contextOrder, contextOffset)) {
			final float f = getVal(hash, array);
			if (outputPrefix == null) return f;
			final long outputOrderAndOffset = getLong(hash, OUTPUT_CONTEXT_OFFSET, array);
			if (outputOrderAndOffset >= 0) {
				outputPrefix.order = orderOf(outputOrderAndOffset);
				outputPrefix.offset = offsetOf(outputOrderAndOffset);
				return f;
			}

		}
		return Float.NaN;
	}

	@Override
	public void putCached(final long contextOffset, final int contextOrder, final int word, final float score, final int hash,
		@OutputParameter final LmContextInfo outputPrefix) {
		final long[] array = !threadSafe ? threadUnsafeArray : threadSafeArray.get();
		setWordAndVal(hash, word, score, array);
		setOutputContextOrderAndOffset(hash, outputPrefix == null ? -1 : outputPrefix.order, outputPrefix == null ? -1 : outputPrefix.offset, array);
		setContextOrderAndOffset(hash, contextOrder, contextOffset, array);

	}

	private static long offsetOf(final long key) {
		return (key & OFFSET_BIT_MASK);
	}

	/**
	 * @param key
	 * @return
	 */
	private static int orderOf(final long key) {
		return (int) ((key & ORDER_BIT_MASK) >>> (NUM_OFFSETS_BITS));
	}

	private int getWord(final int hash, long[] array) {
		return (int) ((array[startOfStruct(hash) + VAL_AND_WORD_OFFSET] & WORD_MASK) >>> Integer.SIZE);
	}

	/**
	 * @param hash
	 * @param off
	 * @return
	 */
	private long getLong(final int hash, final int off, long[] array) {
		return array[startOfStruct(hash) + off];
	}

	private float getVal(final int hash, long[] array) {
		return Float.intBitsToFloat((int) array[startOfStruct(hash) + VAL_AND_WORD_OFFSET]);
	}

	private void setWordAndVal(final int hash, final int word, final float val, long[] array) {
		final long together = combineWordAndVal(word, val);
		array[startOfStruct(hash) + VAL_AND_WORD_OFFSET] = together;
	}

	private long combineWordAndVal(int word, float val) {
		return (((long) word) << Integer.SIZE) | (Float.floatToIntBits(val) & FLOAT_MASK);
	}

	private void setContextOrderAndOffset(final int hash, final int order, final long offset, long[] array) {
		final long together = combine(order, offset);
		setLong(hash, together, CONTEXT_OFFSET, array);
	}

	private void setOutputContextOrderAndOffset(final int hash, final int order, final long offset, long[] array) {
		final long together = combine(order, offset);
		setLong(hash, together, OUTPUT_CONTEXT_OFFSET, array);
	}

	private static long combine(final int order, final long offset) {
		return (((long) order) << (NUM_OFFSETS_BITS)) | offset;
	}

	/**
	 * @param hash
	 * @param l
	 * @param off
	 */
	private void setLong(final int hash, final long l, final int off, long[] array) {
		array[startOfStruct(hash) + off] = l;
	}

	private static int startOfStruct(final int hash) {
		return hash * STRUCT_LENGTH;
	}

	@Override
	public int capacity() {
		return cacheSize;
	}
}
