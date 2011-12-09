package edu.berkeley.nlp.lm.collections;

import java.io.Serializable;
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
public final class LongHashSet implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private long[] keys;

	private int size = 0;

	private static final int EMPTY_KEY = -1;

	private double maxLoadFactor = 0.5;

	public LongHashSet() {
		this(5);
	}

	public void setLoadFactor(double loadFactor) {
		this.maxLoadFactor = loadFactor;
		ensureCapacity(keys.length);
	}

	public LongHashSet(int initCapacity_) {
		int initCapacity = toSize(initCapacity_);
		keys = new long[initCapacity];

		Arrays.fill(keys, EMPTY_KEY);
	}

	public String toString() {
		return Arrays.toString(keys);
	}

	/**
	 * @param initCapacity_
	 * @return
	 */
	private int toSize(int initCapacity_) {
		return Math.max(5, (int) (initCapacity_ / maxLoadFactor) + 1);
	}

	public boolean put(long k) {

		if (size / (double) keys.length > maxLoadFactor) {
			rehash();
		}
		return putHelp(k, keys);

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

		long[] newKeys = new long[length];

		Arrays.fill(newKeys, EMPTY_KEY);
		size = 0;
		for (int i = 0; i < keys.length; ++i) {
			long curr = keys[i];
			if (curr != EMPTY_KEY) {

				putHelp(curr, newKeys);
			}
		}
		keys = newKeys;
	}

	/**
	 * @param k
	 * @param v
	 */
	private boolean putHelp(long k, long[] keyArray) {
		assert k != EMPTY_KEY;

		int pos = find(k, true, keyArray);
		//		int pos = getInitialPos(k, keyArray);
		long currKey = keyArray[pos];
		//		while (currKey != EMPTY_KEY && currKey != k) {
		//			pos++;
		//			if (pos == keyArray.length) pos = 0;
		//			currKey = keyArray[pos];
		//		}
		//

		if (currKey == EMPTY_KEY) {
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
	private static int getInitialPos(final long k, final long[] keyArray) {
		long hash = MurmurHash.hashOneLong(k, 47);
		if (hash < 0) hash = -hash;
		int pos = (int) (hash % keyArray.length);
		return pos;
	}

	public boolean get(long k) {
		int pos = find(k, false);
		return (pos != EMPTY_KEY);

	}

	public boolean containsKey(long k) {
		int pos = find(k, false);
		return (pos != EMPTY_KEY);
	}

	private int find(long k, boolean returnLastEmpty) {
		return find(k, returnLastEmpty, keys);
	}

	/**
	 * @param k
	 * @return
	 */
	private int find(long k, boolean returnLastEmpty, long[] keyArray) {

		final long[] localKeys = keyArray;
		final int length = localKeys.length;
		int pos = getInitialPos(k, localKeys);
		long curr = localKeys[pos];
		while (curr != EMPTY_KEY && curr != k) {
			pos++;
			if (pos == length) pos = 0;
			curr = localKeys[pos];
		}
		return returnLastEmpty ? pos : (curr == EMPTY_KEY ? EMPTY_KEY : pos);

	}

	public boolean isEmpty() {
		return size == 0;
	}

	public void ensureCapacity(int capacity) {
		int newSize = toSize(capacity);
		if (newSize > keys.length) {
			rehash(newSize);
		}
	}

	public int size() {
		return size;
	}

	public void clear() {
		Arrays.fill(keys, EMPTY_KEY);
		size = 0;

	}

	public void remove(long k) {

		int pos = find(k, false, keys);
		if (pos == EMPTY_KEY) return;
		keys[pos] = EMPTY_KEY;

		size--;

	}

	public LongHashSet copy() {
		LongHashSet ret = new LongHashSet();
		ret.keys = Arrays.copyOf(keys, keys.length);
		ret.size = size;
		ret.maxLoadFactor = maxLoadFactor;
		return ret;
	}

}
