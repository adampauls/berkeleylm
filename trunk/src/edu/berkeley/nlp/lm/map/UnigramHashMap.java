package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Iterator;

import edu.berkeley.nlp.lm.collections.Iterators;

/**
 * Low-level hash map implementation which is actually just an array (used for
 * unigrams)
 * 
 * @author adampauls
 * 
 */
final class UnigramHashMap implements Serializable, HashMap
{

	public static long EMPTY_KEY = -1;

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final long numWords;

	private final AbstractNgramMap<?> ngramMap;

	public UnigramHashMap(final long numWords, final AbstractNgramMap<?> ngramMap) {
		this.numWords = numWords;
		this.ngramMap = ngramMap;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#put(long)
	 */
	@Override
	public long put(final long key) {
		return ngramMap.wordOf(key);
	}

	@Override
	public final long getOffset(final long key) {
		final long word = ngramMap.wordOf(key);
		return (word < 0 || word >= numWords) ? EMPTY_KEY : word;
	}

	@Override
	public long getKey(final long contextOffset) {
		return ngramMap.combineToKey((int) contextOffset, 0L);
	}

	@Override
	public boolean isEmptyKey(final long key) {
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

		public RangeIterator(final long numWords) {
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
	public boolean hasContexts(final int word) {
		return (word >= 0 || word < numWords);
	}

}