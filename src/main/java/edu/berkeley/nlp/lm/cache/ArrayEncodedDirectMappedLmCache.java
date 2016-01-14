package edu.berkeley.nlp.lm.cache;

import java.util.Arrays;

/**
 * A direct-mapped cache. This cache does not perform any collision resolution,
 * but rather retains only the most recent key which gets hashed to a particular
 * bucket.
 * 
 * @author adampauls
 * 
 */
public final class ArrayEncodedDirectMappedLmCache implements ArrayEncodedLmCache
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static final int WORD_OFFSET = 0;

	private static final int VAL_OFFSET = 1;

	private static final int KEY_OFFSET = 2;

	private static final int EMPTY = Integer.MIN_VALUE;

	// for efficiency, this array fakes a struct with fields:
	// int firstWord;
	// int length;
	// float val;
	// int[maxNgramOrder -1] key; 
	private final int[] threadUnsafeArray;

	private final ThreadLocal<int[]> threadSafeArray;

	private final int cacheSize;

	private final int structLength;

	private final boolean threadSafe;

	private final int arrayLength;

	public ArrayEncodedDirectMappedLmCache(final int cacheBits, final int maxNgramOrder, final boolean threadSafe) {
		cacheSize = (1 << cacheBits) - 1;
		this.threadSafe = threadSafe;
		arrayLength = maxNgramOrder - 1;
		this.structLength = (maxNgramOrder + 2);
		if (threadSafe) {
			threadUnsafeArray = null;
			threadSafeArray = new ThreadLocal<int[]>()
			{
				@Override
				protected int[] initialValue() {
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
	private int[] allocCache() {
		final int[] ret = new int[cacheSize * structLength];
		Arrays.fill(ret, EMPTY);
		return ret;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.cache.LmCache#getCached(int[], int, int, int)
	 */
	@Override
	public float getCached(final int[] ngram, final int startPos, final int endPos, final int hash) {
		final int[] arrayHere = !threadSafe ? threadUnsafeArray : threadSafeArray.get();
		if (ngram[endPos - 1] == getWord(hash, arrayHere) && equals(ngram, startPos, endPos, arrayHere, getKeyStart(hash))) //
			return getVal(hash, arrayHere);
		else
			return Float.NaN;
	}

	private boolean equals(final int[] ngram, final int startPos, final int endPos, final int[] cachedNgram, final int cachedNgramStart) {
		boolean all = true;
		for (int i = startPos; i < endPos - 1; ++i) {
			all &= cachedNgram[cachedNgramStart + i - startPos] == ngram[i];
		}
		return all && (endPos - startPos - 1 == arrayLength || cachedNgram[cachedNgramStart + endPos - 1 - startPos] == EMPTY);
	}

	private float getVal(final int hash, final int[] arrayHere) {
		return Float.intBitsToFloat(arrayHere[startOfStruct(hash) + VAL_OFFSET]);
	}

	private float setVal(final int hash, final float f, final int[] arrayHere) {
		return arrayHere[startOfStruct(hash) + VAL_OFFSET] = Float.floatToIntBits(f);
	}

	private float setWord(final int hash, final int word, final int[] arrayHere) {
		return arrayHere[startOfStruct(hash) + WORD_OFFSET] = word;
	}

	private int getWord(final int hash, final int[] arrayHere) {
		return arrayHere[startOfStruct(hash) + WORD_OFFSET];
	}

	private int getKeyStart(final int hash) {
		return startOfStruct(hash) + KEY_OFFSET;
	}

	/**
	 * @param hash
	 * @return
	 */
	private int startOfStruct(final int hash) {
		return hash * structLength;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.cache.LmCache#clear()
	 */
	@Override
	public void clear() {
		Arrays.fill(!threadSafe ? threadUnsafeArray : threadSafeArray.get(), Float.floatToIntBits(Float.NaN));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.cache.LmCache#putCached(int[], int, int,
	 * float, int)
	 */
	@Override
	public void putCached(final int[] ngram, final int startPos, final int endPos, final float f, final int hash) {
		final int[] arrayHere = !threadSafe ? threadUnsafeArray : threadSafeArray.get();
		setVal(hash, f, arrayHere);
		setWord(hash, ngram[endPos - 1], arrayHere);
		for (int i = startPos; i < endPos - 1; ++i) {
			arrayHere[getKeyStart(hash) + i - startPos] = ngram[i];
		}
		for (int i = endPos - 1; i < arrayLength; ++i) {
			arrayHere[getKeyStart(hash) + i - startPos] = EMPTY;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.cache.LmCache#size()
	 */
	@Override
	public int capacity() {
		return cacheSize;
	}
}