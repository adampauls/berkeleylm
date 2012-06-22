package edu.berkeley.nlp.lm.bits;

public class BitUtils
{

	private static final long INT_BITS_MASK = ((1L << Integer.SIZE) - 1);

	public static int getHighInt(long key) {
		return ((int) (key >>> Integer.SIZE));

	}

	public static int getLowInt(long key) {
		return (int) (key & INT_BITS_MASK);

	}

	public static long setLowInt(long key, int i) {
		return combineInts(getHighInt(key), i);

	}

	public static long setHighInt(long key, int i) {
		return combineInts(i, getLowInt(key));

	}

	public static long combineInts(int highInt, int lowInt) {
		return (((long) highInt) << Integer.SIZE) | (lowInt & INT_BITS_MASK);
	}

}
