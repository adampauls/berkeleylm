package edu.berkeley.nlp.lm.encoding;

import java.io.Serializable;

import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;

public interface BitCompressor extends Serializable
{

	public BitList compress(long n);

	public long decompress(BitStream bits);

}
