package edu.berkeley.nlp.lm.map;

import java.util.Iterator;

import edu.berkeley.nlp.lm.array.LongArray;

interface HashMap
{

	public static long EMPTY_KEY = -1;

	public long put(final long key);

	public long getOffset(final long key);

	public double getLoadFactor();

	public long getCapacity();

	public long getKey(long contextOffset);

	public boolean isEmptyKey(long key);

	public long size();

	public Iterable<Long> keys();

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

	public boolean hasContexts(int word);

}