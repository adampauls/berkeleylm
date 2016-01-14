package edu.berkeley.nlp.lm.array;

import java.io.ObjectStreamException;
import java.io.Serializable;
import java.util.Arrays;

public final class LongArray implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = -9133624434714616987L;

	/**
	 * For some reason, some VMs will refuse to allocate long[] arrays of size
	 * Integer.MAX_VALUE, so we stick to at most MAX_VALUE / 2
	 */
	private static final int MAX_ARRAY_BITS = 30;

	private static final int MAX_ARRAY_SIZE = 1 << MAX_ARRAY_BITS;

	long size;

	long[][] data;

	// keep a reference to the lowest order array, since that will be used most often
	private long[] first;

	public LongArray(final long initialCapacity) {
		this.size = 0;
		allocFor(initialCapacity, null);
	}

	public static final class StaticMethods
	{
		public static LongArray newLongArray(final long maxKeySize, final long maxNumKeys) {
			return newLongArray(maxKeySize, maxNumKeys, 10);
		}

		public static LongArray newLongArray(@SuppressWarnings("unused") final long maxKeySize, @SuppressWarnings("unused") final long maxNumKeys,
			final long initCapacity) {
			return new LongArray(initCapacity);
		}
	}

	/**
	 * @param capacity
	 */
	private void allocFor(final long capacity, final long[][] old) {
		final int numOuter = o(capacity) + 1;
		final int numInner = i(capacity);
		this.data = new long[numOuter][];
		for (int i = 0; i < numOuter; ++i) {
			final int currSize = (i == numOuter - 1) ? numInner : MAX_ARRAY_SIZE;
			if (old == null || i >= old.length) {
				data[i] = new long[currSize];
			} else if (currSize == old[i].length) {
				data[i] = old[i];
			} else {
				data[i] = Arrays.copyOf(old[i], currSize);
			}
			if (i == 0) first = data[i];
		}
	}

	protected static final int o(final long l) {
		return (int) (l >>> MAX_ARRAY_BITS);
	}

	protected static final int i(final long l) {
		return (int) (l & (MAX_ARRAY_SIZE - 1));
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
		data[o(pos)][i(pos)] = val;
	}

	private void incrementHelp(final long pos, final long val) {
		data[o(pos)][i(pos)] += val;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * edu.berkeley.nlp.mt.lm.util.collections.LongArray#setAndGrowIfNeeeded
	 * (long, long)
	 */
	public void setAndGrowIfNeeded(final long pos, final long val) {
		ensureCapacity(pos + 1);
		setGrowHelp(pos, val);
	}

	public void setAndGrowIfNeededFill(final long pos, final long val) {
		ensureCapacity(pos + 1);
		final long oldSize = size;
		size = Math.max(size, pos + 1);
		for (long i = oldSize; i < size; ++i)
			setHelp(i, val);
	}

	/**
	 * @param pos
	 * @param val
	 */
	private void setGrowHelp(final long pos, final long val) {
		size = Math.max(size, pos + 1);
		setHelp(pos, val);

	}

	public void ensureCapacity(final long minCapacity) {
		final long oldCapacity = sizeOf(data);
		if (minCapacity > oldCapacity) {
			final long[][] oldData = data;
			long newCapacity = (oldCapacity * 3) / 2 + 1;
			if (newCapacity < minCapacity) newCapacity = minCapacity;

			allocFor(newCapacity, oldData);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#get(long)
	 */
	public long get(final long pos) {
		assert (pos < size) : this.getClass().getName() + " array index out of bounds";
		return getHelp(pos);
	}

	private static long sizeOf(final long[][] a) {
		long ret = 0;
		for (int i = 0; i < a.length; ++i) {
			ret += a[i].length;
		}
		return ret;
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
		final int i = i(pos);
		final int o = o(pos);
		return o == 0 ? first[i] : data[o][i];
	}

	public static void main(final String[] argv) {

		long mem = Runtime.getRuntime().maxMemory();
		System.out.println("VM size is " + mem);
		long pos = -9L + Integer.MAX_VALUE / 2;//;4L + Integer.MAX_VALUE;
		final LongArray b = new LongArray(pos + 1);
		final long val = 10000000000000L;
		b.setAndGrowIfNeeded(pos, val);
		final long z = b.get(pos);
		assert z == val;
		System.out.println("Finished with value " + z + " for " + val);
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
		setAndGrowIfNeeded(size, val);
		return true;
	}

	public boolean addWithFixedCapacity(final long val) {
		setGrowHelp(size, val);
		return true;
	}

	public void shift(final long src, final long dest, final int length) {
		if (length == 0) return;
		if (src == dest) return;
		assert dest >= src;

		final int oStart = o(src);
		final int oEnd = o(src + length);
		final int oDestStart = o(dest);
		final int oDestEnd = o(dest + length);
		if (dest + length >= size) {
			setAndGrowIfNeeded(dest + length, 0);
		}
		if (oStart != oEnd || oDestStart != oDestEnd) {
			for (long i = length - 1; i >= 0; --i) {
				set(dest + i, get(src + i));
			}
		} else {
			System.arraycopy(data[o(src)], i(src), data[o(dest)], i(dest), length);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#trimToSize(long)
	 */
	public void trimToSize(final long size_) {
		this.size = size_;
		allocFor(size_, data);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.util.collections.LongArray#fill(long, long)
	 */
	public void fill(final long l, final long n) {
		for (long i = 0; i < n; ++i)
			setAndGrowIfNeeded(i, l);
	}

	public long linearSearch(final long key, final long rangeStart, final long rangeEnd, final long startIndex, final long emptyKey,
		final boolean returnFirstEmptyIndex) {
		for (long i = startIndex; i < rangeEnd; ++i) {
			final long searchKey = getHelp(i);
			if (searchKey == key) return i;
			if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;
		}
		for (long i = rangeStart; i < startIndex; ++i) {
			final long searchKey = getHelp(i);
			if (searchKey == key) return i;
			if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;
		}
		return -1L;
	}

	public void incrementCount(final long index, final long count) {
		if (index >= size) {
			setAndGrowIfNeeded(index, count);
		} else {
			incrementHelp(index, count);
		}
	}

}
