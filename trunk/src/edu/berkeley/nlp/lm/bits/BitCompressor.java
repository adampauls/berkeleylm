package edu.berkeley.nlp.lm.bits;

import java.io.Serializable;


public interface BitCompressor extends Serializable
{

	public BitList compress(long bits);

	public long decompress(BitStream bits);

}
