package edu.berkeley.nlp.lm.bits;

import java.util.BitSet;

public final class BitList
{

	private final BitSet data;

	private int numBits;

	public BitList() {
		data = new BitSet();
		numBits = 0;
	}

	public void add(final boolean b) {
		set(numBits, b);
	}

	public boolean get(final int i) {
		if (i >= size()) throw new IndexOutOfBoundsException();
		return data.get(i);
	}

	public int size() {
		return numBits;
	}

	private void set(final int index, final boolean b) {
		data.set(index, b);
		numBits = Math.max(numBits, index + 1);
	}

	@Override
	public String toString() {
		final StringBuilder s = new StringBuilder("");
		long l = 0L;
		String hex = "";
		for (int i = 0; i < numBits; ++i) {
			final boolean curr = data.get(i);
			if (i % Long.SIZE == 0 && i > 0) {
				hex += String.format("%x", l);
				l = 0;
			}
			l = (l << 1) | (curr ? 1 : 0);
			s.append(curr ? "1" : "0");
		}
		return s.toString() + "(" + hex + ")";
	}

	public void addAll(final BitList bits) {
		for (int i = 0; i < bits.size(); ++i)
			add(bits.get(i));
	}

	public void addLong(final long l) {
		final int size = Long.SIZE;
		addHelp(l, size);
	}

	/**
	 * @param l
	 * @param size
	 */
	private void addHelp(final long l, final int size) {
		for (int b = size - 1; b >= 0; --b) {
			add((l & (1L << b)) != 0);
		}
	}

	public void addShort(final short l) {
		addHelp(l, Short.SIZE);
	}

	public void clear() {
		numBits = 0;
	}

}