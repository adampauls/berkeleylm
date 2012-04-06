package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Iterator;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
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

	private final long keysSize;

	private long numFilled = 0;

	private static final int EMPTY_KEY = -1;

	public ExplicitWordHashMap(final long capacity) {
		keys = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, capacity, capacity);
		keys.fill(EMPTY_KEY, capacity);
		this.keysSize = keys.size();
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
		final long rangeEnd = keysSize;
		final long i = keys.linearSearch(key, rangeStart, rangeEnd, hash, EMPTY_KEY, true);
		if (keys.get(i) == EMPTY_KEY) {
			numFilled++;
			if (numFilled >= keysSize) { throw new RuntimeException("Hash map is full with " + keysSize + " keys. Should never happen."); }
		}
		setKey(i, key);

		return i;
	}

	private void setKey(final long index, final long putKey) {
		keys.set(index, putKey);

	}

	@Override
	public final long getOffset(final long key) {
		final long hash = hash(key);
		if (hash < 0) return -1L;
		final long rangeStart = 0;
		final long rangeEnd = keysSize;
		final long startIndex = hash;
		assert startIndex >= rangeStart;
		assert startIndex < rangeEnd;
		return keys.linearSearch(key, rangeStart, rangeEnd, startIndex, EMPTY_KEY, false);
	}

	@Override
	public long getCapacity() {
		return keysSize;
	}

	@Override
	public double getLoadFactor() {
		return (double) numFilled / getCapacity();
	}

	public double getLoadFactor(int numAdditional) {
		return (double) (numFilled + numAdditional) / getCapacity();
	}

	private long hash(final long key) {
		final long hashed = (MurmurHash.hashOneLong(key, 31));
		long hash1 = hashed;
		if (hash1 < 0) hash1 = -hash1;

		final long startOfRange = 0;
		final long numHashPositions = keysSize - startOfRange;
		if (numHashPositions == 0) return -1;
		hash1 = (hash1 % numHashPositions);
		return hash1 + startOfRange;
	}

	@Override
	public long getKey(final long contextOffset) {
		return keys.get(contextOffset);
	}

	@Override
	public boolean isEmptyKey(final long key) {
		return key == EMPTY_KEY;
	}

	@Override
	public long size() {
		return numFilled;
	}

	public static class KeyIterator implements Iterator<Long>
	{
		private final LongArray keys;

		public KeyIterator(final LongArray keys) {
			this.keys = keys;
			end = keys.size();
			next = -1;
			nextIndex();
		}

		@Override
		public boolean hasNext() {
			return end > 0 && next < end;
		}

		@Override
		public Long next() {
			final long nextIndex = nextIndex();
			return nextIndex;
		}

		long nextIndex() {
			final long curr = next;
			do {
				next++;
			} while (next < end && keys != null && keys.get(next) == EMPTY_KEY);
			return curr;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

		private long next;

		private final long end;
	}

	@Override
	public Iterable<Long> keys() {
		return Iterators.able(new KeyIterator(keys));
	}

	@Override
	public boolean hasContexts(final int word) {
		return true;
	}

}