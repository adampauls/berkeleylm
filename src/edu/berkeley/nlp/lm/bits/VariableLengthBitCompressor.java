package edu.berkeley.nlp.lm.bits;

public class VariableLengthBitCompressor implements BitCompressor
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final int radix;

	public VariableLengthBitCompressor(final int radix) {
		this.radix = radix;
	}

	@Override
	public BitList compress(final long n) {
		return CompressionUtils.variableCompress(n, radix);
	}

	@Override
	public long decompress(final BitStream bits) {
		return CompressionUtils.variableDecompress(bits, radix);
	}

}
