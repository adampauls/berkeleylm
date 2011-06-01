package edu.berkeley.nlp.lm.array;

public interface LongArray
{

	public abstract void set(long pos, long val);

	public abstract void setAndGrowIfNeeded(long pos, long val);

	public abstract long get(long pos);

	public abstract void trim();

	public abstract long size();

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
				if (maxKeySize <= Integer.MAX_VALUE) {
					return new IntSmallLongArray(initCapacity);
				} else {
					return new SmallLongArray(initCapacity);
				}
			} else {
				return new LargeLongArray(initCapacity);
			}
		}

		public static long linearSearch(LongArray array, long key, long rangeStart, long rangeEnd, long startIndex, long emptyKey, boolean returnFirstEmptyIndex) {
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

		public static void incrementCount(LongArray array, long index, long count) {

			if (index >= array.size()) {
				array.setAndGrowIfNeeded(index, count);
			} else {
				long l = array.get(index);
				array.set(index, l + count);
			}
		}
	}

	public abstract long linearSearch(long key, long rangeStart, long rangeEnd, long startIndex, long emptyKey, boolean returnFirstEmptyIndex);

	public abstract void ensureCapacity(long l);

	public abstract void incrementCount(long index, long count);

}