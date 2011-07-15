package edu.berkeley.nlp.lm.array;

public interface LongArray
{

	public abstract void set(long pos, long val);

	public abstract void setAndGrowIfNeeded(long pos, long val);

	public abstract long get(long pos);

	public abstract void trim();

	public abstract long size();

	public abstract boolean addWithFixedCapacity(long val);

	public abstract boolean add(long val);

	public abstract void trimToSize(long size);

	public abstract void fill(long l, long initialCapacity);

	public static final class StaticMethods
	{
		public static LongArray newLongArray(final long maxKeySize, final long maxNumKeys) {
			return newLongArray(maxKeySize, maxNumKeys, 10);
		}

		public static LongArray newLongArray(final long maxKeySize, final long maxNumKeys, final long initCapacity) {
			if (maxNumKeys <= Integer.MAX_VALUE) {
				if (maxKeySize <= Byte.MAX_VALUE) {
					return new ByteLongArray(initCapacity);
				} else if (maxKeySize <= Integer.MAX_VALUE) {
					return new IntLongArray(initCapacity);
				} else {
					return new LongLongArray(initCapacity);
				}
			} else {
				return new LargeLongArray(initCapacity);
			}
		}

		public static long linearSearch(final LongArray array, final long key, final long rangeStart, final long rangeEnd, final long startIndex,
			final long emptyKey, final boolean returnFirstEmptyIndex) {
			long i = startIndex;
			boolean goneAroundOnce = false;
			while (true) {
				if (i == rangeEnd) {
					if (goneAroundOnce) return -1L;
					i = rangeStart;
					goneAroundOnce = true;
				}
				final long searchKey = array.get(i);
				if (searchKey == key) return i;
				if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;
				++i;
			}
		}

		public static void incrementCount(final LongArray array, final long index, final long count) {

			if (index >= array.size()) {
				array.setAndGrowIfNeeded(index, count);
			} else {
				final long l = array.get(index);
				array.set(index, l + count);
			}
		}
	}

	public abstract long linearSearch(long key, long rangeStart, long rangeEnd, long startIndex, long emptyKey, boolean returnFirstEmptyIndex);

	public abstract void ensureCapacity(long l);

	public abstract void incrementCount(long index, long count);

}