package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.bits.BitUtils;
import edu.berkeley.nlp.lm.collections.LongRepresentable;

public class ProbBackoffPair implements Comparable<ProbBackoffPair>, LongRepresentable<ProbBackoffPair>
{

	static final int MANTISSA_MASK = 0x7fffff;

	static final int REST_MASK = ~MANTISSA_MASK;

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + Float.floatToIntBits(prob);
		result = prime * result + Float.floatToIntBits(backoff);
		return result;
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (getClass() != obj.getClass()) return false;
		final ProbBackoffPair other = (ProbBackoffPair) obj;
		if (Float.floatToIntBits(prob) != Float.floatToIntBits(other.prob)) return false;
		if (Float.floatToIntBits(backoff) != Float.floatToIntBits(other.backoff)) return false;
		return true;
	}

	public ProbBackoffPair(final long probBackoff) {
		this(probOf(probBackoff), backoffOf(probBackoff));
	}

	public ProbBackoffPair(final float logProb, final float backoff) {
		this.prob = round(logProb, ConfigOptions.roundBits);
		this.backoff = round(backoff, ConfigOptions.roundBits);
	}

	private float round(final float f, final int mantissaBits) {
		if (Float.isInfinite(f)) return f;
		final int bits = Float.floatToIntBits(f);

		final int mantissa = bits & MANTISSA_MASK;
		final int rest = bits & REST_MASK;
		final int highestBit = Integer.highestOneBit(mantissa);
		int mask = highestBit;
		for (int i = 0; i < mantissaBits; ++i) {
			mask >>>= 1;
			mask |= highestBit;
		}
		final int maskedMantissa = mantissa & mask;
		final float newFloat = Float.intBitsToFloat(rest | maskedMantissa);
		assert Float.isNaN(f) || (Math.abs(f - newFloat) <= 1e-3f) : "Rounding went bad for float " + f + " and rounded " + newFloat;
		return newFloat;
	}

	@Override
	public String toString() {
		return "[FloatPair first=" + prob + ", second=" + backoff + "]";
	}

	public float prob;

	public float backoff;

	@Override
	public int compareTo(final ProbBackoffPair arg0) {
		final int c = Float.compare(prob, arg0.prob);
		if (c != 0) return c;
		return Float.compare(backoff, arg0.backoff);
	}

	@Override
	public long asLong() {
		return floatsToLong(prob, backoff);

	}

	/**
	 * @param prob
	 * @param backoff
	 * @return
	 */
	public static long floatsToLong(final float prob, final float backoff) {
		final int probBits = Float.floatToIntBits(prob);
		final int backoffBits = Float.floatToIntBits(backoff);
		return BitUtils.combineInts(probBits, backoffBits);
	}

	public static float probOf(long key) {
		return Float.intBitsToFloat(BitUtils.getLowInt(key));

	}

	public static float backoffOf(long key) {
		return Float.intBitsToFloat(BitUtils.getHighInt(key));

	}
}