package edu.berkeley.nlp.lm.bits;

public class BitUtils
{

	private static final long INT_BITS_MASK = ((1L << Integer.SIZE) - 1);

	public static int getHighInt(long key) {
		return ((int) (key >>> Integer.SIZE));

	}

	public static int abs(int a_) {
		int a = a_;
		int signMask = a >> (Integer.SIZE - 1); // make a mask of the sign bit
		a ^= signMask; // toggle the bits if value is negative
		a += signMask & 1;
		return a;
	}

	public static long abs(long a_) {
		long a = a_;
		long signMask = a >> (Long.SIZE - 1); // make a mask of the sign bit
		a ^= signMask; // toggle the bits if value is negative
		a += signMask & 1;
		return a;
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

	public static long max(long a_, long b) {
		long a = a_;
		a -= b;
		a &= (~a) >> Long.SIZE - 1;
		a += b;
		return a;
	}

	public static long min(long a_, long b) {
		long a = a_;
		a -= b;
		a &= a >> Long.SIZE - 1;
		a += b;
		return a;
	}

	public static long combineInts(int lowInt, int highInt) {
		return (((long) highInt) << Integer.SIZE) | (lowInt & INT_BITS_MASK);
	}

	public static void main(String[] argv) {
		long x = 0xfff;
		x <<= Long.SIZE - 1;
		x >>>= Long.SIZE - 1;
		System.out.println(x);
	}

}
