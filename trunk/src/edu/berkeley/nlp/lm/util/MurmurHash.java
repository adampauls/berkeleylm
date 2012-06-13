package edu.berkeley.nlp.lm.util;

/**
 * Taken/modified from
 * http://d3s.mff.cuni.cz/~holub/sw/javamurmurhash/MurmurHash.java
 * 
 */
public final class MurmurHash
{

	/**
	 * Generates 32 bit hash from byte array of the given length and seed.
	 * 
	 * @param data
	 *            int array to hash
	 * @param length
	 *            length of the array to hash
	 * @param seed
	 *            initial seed value
	 * @return 32 bit hash of the given array
	 */
	public static int hash32(final int[] data, int startPos, int endPos, int seed) {
		// 'm' and 'r' are mixing constants generated offline.
		// They're not really 'magic', they just happen to work well.
		final int m = 0x5bd1e995;
		final int r = 24;
		final int length = endPos - startPos;
		// Initialize the hash to a random value
		int h = seed ^ length;

		for (int i = startPos; i < endPos; i++) {
			int k = data[i];
			k *= m;
			k ^= k >>> r;
			k *= m;
			h *= m;
			h ^= k;
		}

		h ^= h >>> 13;
		h *= m;
		h ^= h >>> 15;

		return h;
	}

	public static int hash32(final int[] data, int startPos, int endPos) {
		return hash32(data, startPos, endPos, 0x9747b28c);
	}

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

	public static long hashThreeLongs(final long k1, final long k2, final long k3) {

		final int seed = 0x9747b28c;
		final long m = 0xc6a4a7935bd1e995L;
		final int r = 47;

		long h = (seed & 0xffffffffl) ^ (1 * m);
		for (int i = 0; i <= 2; ++i) {
			long k = -1;
			switch (i) {
				case 0:
					k = k1;
					break;
				case 1:
					k = k2;
					break;
				case 2:
					k = k3;
					break;
				default:
					assert false;
			}
			k *= m;
			k ^= k >>> r;
			k *= m;

			h ^= k;
			h *= m;
		}

		h ^= h >>> r;
		h *= m;
		h ^= h >>> r;

		return h;
	}

}