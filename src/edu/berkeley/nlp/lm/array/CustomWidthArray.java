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
public final class CustomWidthArray implements Serializable
{

	public int getKeyWidth() {
		return keyWidth;
	}

	private static final long serialVersionUID = 1L;

	private final static int LOG2_BITS_PER_WORD = 6;

	private final static int BITS_PER_WORD = 1 << LOG2_BITS_PER_WORD;

	private final static int WORD_MASK = BITS_PER_WORD - 1;

	private long size;

	private final int keyWidth;

	private final int fullWidth;

	final long widthDiff;

	private final LongArray data;

	private final static long numLongs(final long size) {
		return ((size + WORD_MASK) >>> LOG2_BITS_PER_WORD);
	}

	private final static long word(final long index) {
		return (index >>> LOG2_BITS_PER_WORD);
	}

	private final static long bit(final long index) {
		return (index & WORD_MASK);
	}

	private final static long mask(final long index) {
		return 1L << (index & WORD_MASK);
	}

	public CustomWidthArray(final long numWords, final int keyWidth) {
		this(numWords, keyWidth, keyWidth);
	}

	public CustomWidthArray(final long numWords, final int keyWidth, final int fullWidth) {
		assert keyWidth > 0;
		assert fullWidth > 0;
		this.keyWidth = keyWidth;
		this.fullWidth = fullWidth;
		this.widthDiff = Long.SIZE - keyWidth;
		final long numBits = numWords * fullWidth;
		data = new LongArray(numLongs(numBits));// new long[numLongs(numBits)];
		size = 0;
	}

	private long length() {
		return size;
	}

	public void ensureCapacity(final long numWords) {
		final long numBits = numWords * fullWidth;
		final long numLongs = numLongs(numBits);
		data.ensureCapacity(numLongs);
		if (numLongs > data.size()) data.setAndGrowIfNeeded(numLongs - 1, 0);
	}

	public void trim() {
		trimToSize(size);
	}

	/**
	 * @param sizeHere
	 */
	public void trimToSize(final long sizeHere) {
		final long numBits = sizeHere * fullWidth;
		data.trimToSize(numLongs(numBits));
	}

	private void rangeCheck(final long index) {
		if (index >= length()) { //
			throw new IndexOutOfBoundsException("Index (" + index + ") is greater than length (" + (length()) + ")");
		}
	}

	public boolean getBit(final long index) {
		rangeCheck(index);
		return (data.get(word(index)) & mask(index)) != 0;
	}

	public void clear(final long index) {
		rangeCheck(index);
		data.set(word(index), data.get(word(index)) & ~mask(index));
	}

	private long getLong(final long from, final long l) {
		if (l == Long.SIZE) return 0;
		final long startWord = word(from);
		final long startBit = bit(from);
		if (startBit <= l)
			return data.get(startWord) << l - startBit >>> l;
		else
			return data.get(startWord) >>> startBit | data.get(startWord + 1) << Long.SIZE + l - startBit >>> l;
	}

	public boolean add(final long value) {
		return addHelp(value, true);
	}

	public boolean addWithFixedCapacity(final long value) {
		return addHelp(value, false);
	}

	/**
	 * @param value
	 * @return
	 */
	private boolean addHelp(final long value, final boolean growCapacity) {
		assert fullWidth == keyWidth;
		final long length = this.size * fullWidth;
		final long startWord = word(length);
		final long startBit = bit(length);
		if (growCapacity) ensureCapacity(this.size + 1);

		if (startBit + keyWidth <= Long.SIZE)
			data.set(startWord, data.get(startWord) | (value << startBit));
		else {
			data.set(startWord, data.get(startWord) | (value << startBit));
			data.set(startWord + 1, value >>> BITS_PER_WORD - startBit);
		}

		this.size++;
		return true;
	}

	public long get(final long index) {
		return getHelp(index, 0, keyWidth);
	}

	public long get(final long index, int offset, int width) {
		return getHelp(index, offset, width);
	}

	/**
	 * @param index
	 * @return
	 */
	private long getHelp(final long index, int offset, int width) {
		final long start = index * fullWidth + offset;
		return getLong(start, Long.SIZE - width);
	}

	public static int numBitsNeeded(final long n) {
		if (n == 0) return 1;
		if (Long.bitCount(n) == 1)
			return Long.numberOfTrailingZeros(n) + 1;
		else
			return Long.SIZE - Long.numberOfLeadingZeros(n - 1);
	}

	public void set(final long index, final long value) {
		rangeCheck(index);
		final int offset = 0;
		final int width = keyWidth;
		setHelp(index, value, offset, width);
	}

	public void set(final long index, final long value, final int offset, final int width) {
		rangeCheck(index);
		setHelp(index, value, offset, width);
	}

	/**
	 * @param index
	 * @param value
	 * @param offset
	 */
	private void setHelp(final long index, final long value, final int offset, final int width) {

		assert numBitsNeeded(value) <= width : "Value " + value + " bits " + width;
		final long start = index * fullWidth + offset;
		final long startWord = word(start);
		final long endWord = word(start + width - 1);
		final long startBit = bit(start);
		final long fullMask = width == Long.SIZE ? -1L : ((1L << width) - 1);

		if (startWord == endWord) {
			long startWordLong = data.get(startWord);
			startWordLong &= ~(fullMask << startBit);
			startWordLong |= value << startBit;
			data.set(startWord, startWordLong);

			assert value == (startWordLong >>> startBit & fullMask) : startWord + " " + startBit + " " + value;
		} else {
			// Here startBit > 0.
			long startWordLong = data.get(startWord);
			startWordLong &= ((1L << startBit) - 1);
			startWordLong |= (value << startBit);
			data.set(startWord, startWordLong);
			long endWordLong = data.get(endWord);
			endWordLong &= (-(1L << width - BITS_PER_WORD + startBit));
			endWordLong |= (value >>> BITS_PER_WORD - startBit);
			data.set(endWord, endWordLong);

			assert value == (startWordLong >>> startBit | endWordLong << (BITS_PER_WORD - startBit) & fullMask);
		}
	}

	public void setAndGrowIfNeeded(final long pos, final long value) {
		if (pos >= size) {
			ensureCapacity(pos + 2);
			this.size = pos + 1;
		}
		set(pos, value);
	}

	public void setAndGrowIfNeeded(final long pos, final long value, final int offset, final int width) {
		if (pos >= size) {
			ensureCapacity(pos + 2);
			this.size = pos + 1;
		}
		set(pos, value, offset, width);
	}

	public long size() {
		return length();
	}

	public void fill(final long l, final long n) {
		final long numBits = n * fullWidth;
		final long numLongs = numLongs(numBits);
		data.fill(l, numLongs);
		size = Math.max(n, size);
	}

	public long linearSearch(final long key, final long rangeStart, final long rangeEnd, final long startIndex, final long emptyKey,
		final boolean returnFirstEmptyIndex) {
		for (long i = startIndex; i < rangeEnd; ++i) {
			final long searchKey = getHelp(i, 0, keyWidth);
			if (searchKey == key) return i;
			if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;
		}
		for (long i = rangeStart; i < startIndex; ++i) {
			final long searchKey = getHelp(i, 0, keyWidth);
			if (searchKey == key) return i;
			if (searchKey == emptyKey) return returnFirstEmptyIndex ? i : -1L;
		}
		return -1L;
	}

	public void incrementCount(final long index, final long count) {
		if (index >= size()) {
			setAndGrowIfNeeded(index, count);
		} else {
			final long curr = get(index);
			set(index, curr + count);
		}
	}

	public int getFullWidth() {
		return fullWidth;
	}

}
