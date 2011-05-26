package edu.berkeley.nlp.lm.util;

/**
 * Taken from http://d3s.mff.cuni.cz/~holub/sw/javamurmurhash/MurmurHash.java
 * 
 */
public final class MurmurHash
{

	public static long hashOneLong(final long k_, final int seed) {
		long k = k_;
		final long m = 0xc6a4a7935bd1e995L;
		final int r = 47;

		long h = (seed & 0xffffffffl) ^ (1 * m);

		k *= m;
		k ^= k >>> r;
		k *= m;

		h ^= k;
		h *= m;

		h ^= h >>> r;
		h *= m;
		h ^= h >>> r;

		return h;
	}

}