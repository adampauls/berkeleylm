package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Map;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.MurmurHash;

/**
 * Low-level hash map which stored context-encoded parent pointers in a trie.
 * 
 * @author adampauls
 * 
 */
final class ExplicitWordHashMap implements Serializable, HashMap
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@PrintMemoryCount
	private final LongArray keys;

	private long numFilled = 0;

	private static final int EMPTY_KEY = -1;

	public ExplicitWordHashMap(long capacity) {
		keys = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, capacity, capacity);
		keys.fill(EMPTY_KEY, capacity);
		numFilled = 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#put(long)
	 */
	@Override
	public long put(final long key) {
		final long hash = hash(key);
		if (hash < 0) return -1L;
		final long rangeStart = 0;
		final long rangeEnd = keys.size();
		long i = keys.linearSearch(key, rangeStart, rangeEnd, hash, EMPTY_KEY, true);
		if (keys.get(i) == EMPTY_KEY) {
			numFilled++;
		}
		setKey(i, key);

		return i;
	}

	private void setKey(final long index, final long putKey) {
		keys.set(index, putKey);

	}

	public final long getOffset(final long key) {
		final long hash = hash(key);
		if (hash < 0) return -1L;
		final long rangeStart = 0;
		final long rangeEnd = keys.size();
		final long startIndex = hash;
		assert startIndex >= rangeStart;
		assert startIndex < rangeEnd;
		return keys.linearSearch(key, rangeStart, rangeEnd, startIndex, EMPTY_KEY, false);
	}

	public long getCapacity() {
		return keys.size();
	}

	public double getLoadFactor() {
		return (double) numFilled / getCapacity();
	}

	private long hash(final long key) {
		final long hashed = (MurmurHash.hashOneLong(key, 31));
		long hash1 = hashed;
		if (hash1 < 0) hash1 = -hash1;

		final long startOfRange = 0;
		final long numHashPositions = keys.size() - startOfRange;
		if (numHashPositions == 0) return -1;
		hash1 = (hash1 % numHashPositions);
		return hash1 + startOfRange;
	}

	@Override
	public long getKey(long contextOffset) {
		return keys.get(contextOffset);
	}

	@Override
	public boolean isEmptyKey(long key) {
		return key == EMPTY_KEY;
	}

	@Override
	public long size() {
		return numFilled;
	}

	@Override
	public Iterable<Long> keys() {
		return Iterators.able(new KeyIterator(keys));
	}

}