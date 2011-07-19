package edu.berkeley.nlp.lm.array;

import java.io.ObjectStreamException;
import java.io.Serializable;
import java.util.Arrays;

public final class IntLongArray implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -9133624434714616987L;

	private long size;

	private int[] data;

	public IntLongArray(final long initialCapacity) {
		this.size = 0;
		allocFor(initialCapacity, null);
	}

	/**
	 * @param capacity
	 */
	private void allocFor(final long capacity, final int[] old) {
		check(capacity);
		final int numInner = i(capacity);
		this.data = old == null ? new int[numInner] : Arrays.copyOf(old, numInner);
	}

	/**
	 * @param capacity
	 */
	private void check(final long capacity) {
		if (capacity >= Integer.MAX_VALUE) throw new IllegalArgumentException(capacity + " to big for " + IntLongArray.class.getSimpleName());
	}

	private static final int i(final long l) {
		return (int) l;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#set(long, long)
	 */

	public void set(final long pos, final long val) {
		if (pos >= size) throw new ArrayIndexOutOfBoundsException("" + pos);
		setHelp(pos, val);

	}

	/**
	 * @param pos
	 * @param val
	 */
	private void setHelp(final long pos, final long val) {
		data[i(pos)] = (int) val;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.berkeley.nlp.mt.lm.util.collections.LongArray#setAndGrowIfNeeeded
	 * (long, long)
	 */

	public void setAndGrowIfNeeded(final long pos, final long val) {
		setGrowHelp(pos, val, true);
	}

	/**
	 * @param pos
	 * @param val
	 */
	private void setGrowHelp(final long pos, final long val, final boolean growCapacity) {
		check(pos);
		if (growCapacity) ensureCapacity(pos + 1);
		size = Math.max(size, pos + 1);
		setHelp(pos, val);
	}

	public void ensureCapacity(final long minCapacity) {
		final int oldCapacity = sizeOf(data);
		if (minCapacity > oldCapacity) {
			final int[] oldData = data;
			int newCapacity = Math.min(Integer.MAX_VALUE, (oldCapacity * 3) / 2 + 1);
			if (newCapacity < minCapacity) newCapacity = (int) minCapacity;

			allocFor(newCapacity, oldData);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#get(long)
	 */

	public long get(final long pos) {
		if (pos >= size) throw new ArrayIndexOutOfBoundsException("" + pos);
		return getHelp(pos);
	}

	private static int sizeOf(final int[] a) {
		return a.length;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#trim()
	 */

	public void trim() {
		allocFor(size, data);
	}

	/**
	 * @param pos
	 * @return
	 */
	private long getHelp(final long pos) {
		return data[i(pos)];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#size()
	 */

	public long size() {
		return size;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#add(long)
	 */

	public boolean add(final long val) {
		setGrowHelp(size, val, true);
		return true;
	}

	public boolean addWithFixedCapacity(final long val) {
		setGrowHelp(size, val, false);
		return true;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#trimToSize(long)
	 */

	public void trimToSize(@SuppressWarnings("hiding") final long size) {
		allocFor(size, data);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#fill(long, long)
	 */

	public void fill(final long l, final long initialCapacity) {
		for (int i = (int) initialCapacity; i >= 0; --i)
			setAndGrowIfNeeded(i, l);
	}

	public long linearSearch(final long key, final long rangeStart, final long rangeEnd, final long startIndex, final long emptyKey,
		final boolean returnFirstEmptyIndex) {
		final int[] localData = data;
		for (int i = (int) startIndex; i < rangeEnd; ++i) {
			final long searchKey = localData[i];
			if (searchKey == key) return i;
			if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;

		}
		for (int i = (int) rangeStart; i < (int) startIndex; ++i) {
			final long searchKey = localData[i];
			if (searchKey == key) return i;
			if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;
		}
		return -1L;
	}

	public void incrementCount(final long index, final long count) {
		if (index >= size()) {
			setAndGrowIfNeeded(index, count);
		} else {
			final long l = get(index);
			set(index, l + count);
		}
	}

	@SuppressWarnings("unused")
	private Object readResolve() throws ObjectStreamException {
		System.gc();
		System.gc();
		System.gc();
		return this;
	}

}
