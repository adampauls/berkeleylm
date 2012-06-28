package edu.berkeley.nlp.lm.bits;

public class BitUtils
{

	private static final long INT_BITS_MASK = ((1L << Integer.SIZE) - 1);

	public static int getHighInt(long key) {
		return ((int) (key >>> Integer.SIZE));

	}

	public static int getLowInt(long key) {
		return (int) getLowLong(key);

	}

	public static long getLowLong(long key) {
		return (key & INT_BITS_MASK);
	}

	public static long setLowInt(long key, int i) {
		return combineInts(i, getHighInt(key));

	}

	public static long setHighInt(long key, int i) {
		return combineInts(getLowInt(key), i);

	}

	public static long combineInts(int lowInt, int highInt) {
		return (((long) highInt) << Integer.SIZE) | (lowInt & INT_BITS_MASK);
	}

}
