package edu.berkeley.nlp.lm.bits;

import java.io.Serializable;

public final class VariableLengthBitCompressor implements Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final int radix;

	public VariableLengthBitCompressor(final int radix) {
		this.radix = radix;
	}

	public BitList compress(final long n) {
		return CompressionUtils.variableCompress(n, radix);
	}

	public long decompress(final BitStream bits) {
		return CompressionUtils.variableDecompress(bits, radix);
	}

}
