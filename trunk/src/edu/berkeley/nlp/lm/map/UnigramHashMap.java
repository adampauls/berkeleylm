package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.MurmurHash;

/**
 * Low-level hash map implementation which is actually just an array (used for
 * unigrams)
 * 
 * @author adampauls
 * 
 */
final class UnigramHashMap implements Serializable, HashMap
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final long numWords;

	public UnigramHashMap(final long numWords) {
		this.numWords = numWords;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#put(long)
	 */
	@Override
	public long put(final long key) {

		return AbstractNgramMap.wordOf(key);
	}

	public final long getOffset(final long key) {
		final long word = AbstractNgramMap.wordOf(key);
		return (word < 0 || word >= numWords) ? EMPTY_KEY : word;
	}

	@Override
	public long getKey(long contextOffset) {
		return AbstractNgramMap.combineToKey((int) contextOffset, 0L);
	}

	@Override
	public boolean isEmptyKey(long key) {
		return key == EMPTY_KEY;
	}

	@Override
	public long size() {
		return numWords;
	}

	@Override
	public Iterable<Long> keys() {
		return Iterators.able(new RangeIterator(numWords));
	}

	@Override
	public double getLoadFactor() {
		return 1.0;
	}

	@Override
	public long getCapacity() {
		return numWords;
	}

	private static class RangeIterator implements Iterator<Long>
	{

		private final long numWords;

		private long i = 0;

		public RangeIterator(long numWords) {
			this.numWords = numWords;
		}

		@Override
		public boolean hasNext() {
			return i < numWords;
		}

		@Override
		public Long next() {
			return i++;
		}

		@Override
		public void remove() {
			// TODO Auto-generated method stub
			throw new UnsupportedOperationException("Method not yet implemented");
		}

	}

	@Override
	public boolean hasContexts(int word) {
		return (word >= 0 || word < numWords);
	}

}