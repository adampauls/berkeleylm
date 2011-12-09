package edu.berkeley.nlp.lm.collections;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.util.MurmurHash;

/**
 * Open address hash map with linear probing. Assumes keys are non-negative
 * (uses -1 internally for empty key). Returns 0.0 for keys not in the map.
 * 
 * @author adampauls
 * 
 */
public final class LongToIntHashMap
{

	private long[] keys;

	private int[] values;

	private int size = 0;

	private static final int EMPTY_VAL = -1;

	private double maxLoadFactor = 0.5;

	private boolean sorted = false;

	//	private int deflt = -1;

	public LongToIntHashMap() {
		this(5);
	}

	public void setLoadFactor(double loadFactor) {
		this.maxLoadFactor = loadFactor;
		ensureCapacity(values.length);
	}

	public LongToIntHashMap(int initCapacity_) {
		int initCapacity = toSize(initCapacity_);
		keys = new long[initCapacity];
		values = new int[initCapacity];
		Arrays.fill(values, EMPTY_VAL);
	}

	public String toString() {
		String s = "[";
		for (Entry entry : primitiveEntries()) {
			s += s.length() == 1 ? "" : " ";
			s += "(" + entry.key + "," + entry.value + ")";
		}
		s += "]";
		return s;
	}

	public void toSorted() {
		sorted = true;
		long[] newKeys = new long[size];
		int[] newValues = new int[size];
		List<Entry> sortedEntries = new ArrayList<Entry>(size);
		for (java.util.Map.Entry<Long, Integer> e : entries()) {
			sortedEntries.add((Entry) e);
		}
		Collections.sort(sortedEntries, new Comparator<Entry>()
		{

			public int compare(Entry o1, Entry o2) {
				return Double.compare(o1.key, o2.key);
			}
		});
		int k = 0;
		for (Entry e : sortedEntries) {
			newKeys[k] = e.getKey();
			newValues[k] = e.getValue();
			k++;
		}
		keys = newKeys;
		values = newValues;
	}

	/**
	 * @param initCapacity_
	 * @return
	 */
	private int toSize(int initCapacity_) {
		return Math.max(5, (int) (initCapacity_ / maxLoadFactor) + 1);
	}

	public void put(Long k, int v) {
		checkNotImmutable();
		if (size / (double) keys.length > maxLoadFactor) {
			rehash();
		}
		putHelp(k, v, keys, values);

	}

	public void incrementCount(long k, int d) {
		checkNotImmutable();
		if (d == 0) return;
		int pos = find(k, false);
		if (pos == EMPTY_VAL || pos == EMPTY_VAL)
			put(k, d);
		else
			values[pos] += d;

	}

	/**
	 * 
	 */
	private void checkNotImmutable() {
		if (keys == null) throw new RuntimeException("Cannot change wrapped IntCounter");
		if (sorted) throw new RuntimeException("Cannot change sorted IntCounter");
	}

	/**
	 * 
	 */
	private void rehash() {
		final int length = keys.length * 2 + 1;
		rehash(length);
	}

	/**
	 * @param length
	 */
	private void rehash(final int length) {
		checkNotImmutable();
		long[] newKeys = new long[length];
		int[] newValues = new int[length];
		Arrays.fill(newValues, EMPTY_VAL);
		size = 0;
		for (int i = 0; i < keys.length; ++i) {
			long curr = keys[i];
			int val = values[i];
			if (val != EMPTY_VAL) {
				putHelp(curr, val, newKeys, newValues);
			}
		}
		keys = newKeys;
		values = newValues;
	}

	/**
	 * @param k
	 * @param v
	 */
	private boolean putHelp(long k, int v, long[] keyArray, int[] valueArray) {
		checkNotImmutable();
		assert v >= 0;
		int pos = find(k, true, keyArray, valueArray);
		//		int pos = getInitialPos(k, keyArray);
		//		long currKey = keyArray[pos];
		//		while (currKey != EMPTY_KEY && currKey != k) {
		//			pos++;
		//			if (pos == keyArray.length) pos = 0;
		//			currKey = keyArray[pos];
		//		}
		//
		boolean wasEmpty = valueArray[pos] == EMPTY_VAL;
		valueArray[pos] = v;
		if (wasEmpty) {
			size++;
			keyArray[pos] = k;
			return true;
		}
		return false;
	}

	/**
	 * @param k
	 * @param keyArray
	 * @return
	 */
	private static int getInitialPos(final long k, final int length) {
		if (length < 0) return (int) k;
		long hash = MurmurHash.hashOneLong(k, 31);
		if (hash < 0) hash = -hash;
		int pos = (int) (hash % length);
		return pos;
	}

	public int get(long k, int def) {
		int pos = find(k, false);
		if (pos == EMPTY_VAL) return def;

		return values[pos];
	}

	private int find(long k, boolean returnLastEmpty) {
		return find(k, returnLastEmpty, keys, values);
	}

	/**
	 * @param k
	 * @return
	 */
	private int find(long k, boolean returnLastEmpty, long[] keyArray, int[] valueArray) {
		if (keyArray == null) {
			return (int) (k < valueArray.length ? k : EMPTY_VAL);
		} else if (sorted) {
			final int pos = Arrays.binarySearch(keyArray, k);
			return pos < 0 ? EMPTY_VAL : pos;

		} else {
			final int[] localValues = valueArray;
			final int length = localValues.length;
			int pos = getInitialPos(k, localValues.length);
			int currVal = localValues[pos];
			long curr = keyArray[pos];
			while (currVal != EMPTY_VAL && curr != k) {
				pos++;
				if (pos == length) pos = 0;
				currVal = localValues[pos];
				curr = keyArray[pos];
			}
			return returnLastEmpty ? pos : (currVal == EMPTY_VAL ? EMPTY_VAL : pos);
		}
	}

	//	public void setDefault(int d) {
	//		this.deflt = d;
	//	}

	public boolean isEmpty() {
		return size == 0;
	}

	public class Entry implements Map.Entry<Long, Integer>, Comparable<Entry>
	{
		private int index;

		public long key;

		public int value;

		public Entry(long key, int value, int index) {
			super();
			this.key = key;
			assert value >= 0;
			this.value = value;
			this.index = index;
		}

		public Long getKey() {
			return key;
		}

		public Integer getValue() {
			return value;
		}

		public Integer setValue(Integer value) {
			this.value = value;
			values[index] = value;
			return this.value;
		}

		@Override
		public int compareTo(Entry o) {
			// sortable by *value*
			return Double.compare(value, o.value);
		}
	}

	private class EntryIterator extends MapIterator<Map.Entry<Long, Integer>>
	{
		public Entry next() {
			final int nextIndex = nextIndex();
			return new Entry(keys == null ? nextIndex : keys[nextIndex], values[nextIndex], nextIndex);
		}
	}

	private class KeyIterator extends MapIterator<Long>
	{
		public Long next() {
			final int nextIndex = nextIndex();
			return keys == null ? nextIndex : keys[nextIndex];
		}
	}

	private class PrimitiveEntryIterator extends MapIterator<Entry>
	{
		public Entry next() {
			final int nextIndex = nextIndex();
			return new Entry(keys == null ? nextIndex : keys[nextIndex], values[nextIndex], nextIndex);
		}
	}

	private abstract class MapIterator<E> implements Iterator<E>
	{
		public MapIterator() {
			end = keys == null ? size : values.length;
			next = -1;
			nextIndex();
		}

		public boolean hasNext() {
			return end > 0 && next < end;
		}

		int nextIndex() {
			int curr = next;
			do {
				next++;
			} while (next < end && keys != null && values[next] == EMPTY_VAL);
			return curr;
		}

		public void remove() {
			throw new UnsupportedOperationException();
		}

		private int next, end;
	}

	public Iterable<Map.Entry<Long, Integer>> entries() {
		return Iterators.able(new EntryIterator());
	}

	public void ensureCapacity(int capacity) {
		checkNotImmutable();
		int newSize = toSize(capacity);
		if (newSize > keys.length) {
			rehash(newSize);
		}
	}

	public int size() {
		return size;
	}

	public Iterable<Entry> primitiveEntries() {
		return new Iterable<Entry>()
		{
			public Iterator<Entry> iterator() {
				return (new PrimitiveEntryIterator());
			}
		};

	}

	public Iterable<Long> keySet() {
		return Iterators.able(new KeyIterator());
	}

	public void clear() {
		//		Arrays.fill(keys, EMPTY_KEY);
		Arrays.fill(values, EMPTY_VAL);
		size = 0;

	}

	public List<Entry> getObjectsSortedByValue(boolean descending) {
		List<edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry> l = new ArrayList<edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry>();
		for (final edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry entry : primitiveEntries()) {
			l.add(entry);
		}
		Collections.sort(l);
		if (descending) Collections.reverse(l);
		return l;
	}

	public LongToIntHashMap copy() {
		LongToIntHashMap ret = new LongToIntHashMap();
		//		ret.deflt = deflt;
		ret.keys = Arrays.copyOf(keys, keys.length);
		ret.values = Arrays.copyOf(values, values.length);
		ret.size = size;
		ret.sorted = sorted;
		ret.maxLoadFactor = maxLoadFactor;
		return ret;
	}

}
