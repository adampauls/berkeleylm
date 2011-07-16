package edu.berkeley.nlp.lm.array;

import java.io.ObjectStreamException;
import java.io.Serializable;

/**
 * An array with a custom word "width" in bits. Borrows heavily from Sux4J
 * (http://sux.dsi.unimi.it/)
 * 
 * @author adampauls
 * 
 */
@SuppressWarnings("ucd")
public final class CustomWidthArray implements LongArray, Serializable
{

	private static final long serialVersionUID = 1L;

	private final static int LOG2_BITS_PER_WORD = 6;

	private final static int BITS_PER_WORD = 1 << LOG2_BITS_PER_WORD;

	private final static int WORD_MASK = BITS_PER_WORD - 1;

	private long size;

	private final int width;

	private final long fullMask;

	private final LongLongArray data;

	private final static long numLongs(final long size) {
		//		assert (size + WORD_MASK) >>> LOG2_BITS_PER_WORD <= Integer.MAX_VALUE;
		return ((size + WORD_MASK) >>> LOG2_BITS_PER_WORD);
	}

	private final static long word(final long index) {
		//		assert index >>> LOG2_BITS_PER_WORD <= Integer.MAX_VALUE;
		return (index >>> LOG2_BITS_PER_WORD);
	}

	private final static long bit(final long index) {
		return (index & WORD_MASK);
	}

	private final static long mask(final long index) {
		return 1L << (index & WORD_MASK);
	}

	public CustomWidthArray(final long numWords, final int width) {
		final long numBits = numWords * width;
		//		assert (numBits <= (Integer.MAX_VALUE + 1L) * Long.SIZE) : ("CustomWidthArray can only be 2^37 bits long");
		data = (LongLongArray) LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, numLongs(numBits));// new long[numLongs(numBits)];
		size = 0;
		this.width = width;
		fullMask = width == Long.SIZE ? -1 : ((1L << width) - 1);
	}

	private long length() {
		return size;
	}

	@Override
	public void ensureCapacity(final long numWords) {
		final long numBits = numWords * width;
		//		assert (numBits <= (Integer.MAX_VALUE + 1L) * Long.SIZE) : ("CustomWidthArray can only be 2^37 bits long");
		final long numLongs = numLongs(numBits);
		data.ensureCapacity(numLongs);
		if (numLongs > data.size()) data.setAndGrowIfNeeded(numLongs - 1, 0);
	}

	@Override
	public void trim() {
		trimToSize(size);
	}

	/**
	 * @param sizeHere
	 */
	@Override
	public void trimToSize(final long sizeHere) {
		final long numBits = sizeHere * width;
		data.trimToSize(numLongs(numBits));
	}

	private void rangeCheck(final long index) {
		if (index >= length()) throw new IndexOutOfBoundsException("Index (" + index + ") is greater than length (" + (length()) + ")");
	}

	public boolean getBit(final long index) {
		rangeCheck(index);
		return (data.get(word(index)) & mask(index)) != 0;
	}

	public boolean set(final long index, final boolean value) {
		rangeCheck(index);
		final long word = word(index);
		final long mask = mask(index);
		final long currVal = data.get(word);
		final boolean oldValue = (currVal & mask) != 0;
		if (value)
			data.set(word, currVal | mask);
		else
			data.set(word, currVal & ~mask);
		return oldValue != value;
	}

	public void set(final long index) {
		rangeCheck(index);
		data.set(word(index), data.get(word(index)) | mask(index));
	}

	public void clear(final long index) {
		rangeCheck(index);
		data.set(word(index), data.get(word(index)) & ~mask(index));
	}

	private long getLong(final long from, final long to) {
		final long l = Long.SIZE - (to - from);
		final long startWord = word(from);
		final long startBit = bit(from);
		if (l == Long.SIZE) return 0;
		if (startBit <= l) return data.get(startWord) << l - startBit >>> l;
		return data.get(startWord) >>> startBit | data.get(startWord + 1) << Long.SIZE + l - startBit >>> l;
	}

	@Override
	public boolean add(final long value) {
		return addHelp(value, true);
	}

	@Override
	public boolean addWithFixedCapacity(final long value) {
		return addHelp(value, false);
	}

	/**
	 * @param value
	 * @return
	 */
	private boolean addHelp(final long value, final boolean growCapacity) {
		assert !(width < Long.SIZE && (value & -1L << width) != 0) : "The specified value (" + value
			+ ") is larger than the maximum value for the given width (" + width + ")";
		final long length = this.size * width;
		final long startWord = word(length);
		final long startBit = bit(length);
		if (growCapacity) ensureCapacity(this.size + 1);

		if (startBit + width <= Long.SIZE)
			data.set(startWord, data.get(startWord) | (value << startBit));
		else {
			data.set(startWord, data.get(startWord) | (value << startBit));
			data.set(startWord + 1, value >>> BITS_PER_WORD - startBit);
		}

		this.size++;
		return true;
	}

	@Override
	public long get(final long index) {
		final long start = index * width;
		return getLong(start, start + width);
	}

	public static int numBitsNeeded(final long n) {
		if (n == 0) return 1;
		final int num = Long.SIZE - Long.numberOfLeadingZeros(n - 1);
		if (n % 2 == 0) return num + 1;
		return num;
	}

	@Override
	public void set(final long index, final long value) {
		rangeCheck(index);
		if (width == 0) return;
		if (width != Long.SIZE && value > fullMask) throw new IllegalArgumentException("Value too large: " + value);
		final long start = index * width;
		final long startWord = word(start);
		final long endWord = word(start + width - 1);
		final long startBit = bit(start);

		if (startWord == endWord) {
			data.set(startWord, data.get(startWord) & ~(fullMask << startBit));
			data.set(startWord, data.get(startWord) | (value << startBit));
			assert value == (data.get(startWord) >>> startBit & fullMask) : startWord + " " + startBit + " " + value;
		} else {
			// Here startBit > 0.
			data.set(startWord, data.get(startWord) & ((1L << startBit) - 1));
			data.set(startWord, data.get(startWord) | (value << startBit));
			data.set(endWord, data.get(endWord) & (-(1L << width - BITS_PER_WORD + startBit)));
			data.set(endWord, data.get(endWord) | (value >>> BITS_PER_WORD - startBit));

			assert value == (data.get(startWord) >>> startBit | data.get(endWord) << (BITS_PER_WORD - startBit) & fullMask);
		}
	}

	@Override
	public void setAndGrowIfNeeded(final long pos, final long value) {
		if (pos >= size) {
			ensureCapacity(pos + 2);
			this.size = pos + 1;
		}
		set(pos, value);
	}

	@Override
	public long size() {
		return length();
	}

	@Override
	public void fill(final long l, final long initialCapacity) {
		final long numBits = initialCapacity * width;
		data.fill(numLongs(numBits), l);
	}

	@Override
	public long linearSearch(final long key, final long rangeStart, final long rangeEnd, final long startIndex, final long emptyKey,
		final boolean returnFirstEmptyIndex) {
		return LongArray.StaticMethods.linearSearch(this, key, rangeStart, rangeEnd, startIndex, emptyKey, returnFirstEmptyIndex);
	}

	@Override
	public void incrementCount(final long index, final long count) {
		LongArray.StaticMethods.incrementCount(this, index, count);
	}

	@SuppressWarnings("unused")
	private Object readResolve() throws ObjectStreamException {
		System.gc();
		System.gc();
		System.gc();
		return this;
	}

}
