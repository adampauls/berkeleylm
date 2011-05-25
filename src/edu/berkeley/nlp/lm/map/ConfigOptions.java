package edu.berkeley.nlp.lm.map;

import java.io.Serializable;

import edu.berkeley.nlp.lm.util.Annotations.Option;

public class ConfigOptions implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	/**
	 * Number of longs (8 bytes) used as a "block" for variable length
	 * compression.
	 */
	@Option(gloss = "Number of longs (8 bytes) used as a block for variable length compression")
	public int compressedBlockSize = 16;

	@Option(gloss = "Parameter \"k\" which controls the base for variable-length compression of offset deltas")
	public int offsetDeltaRadix = 6;

	@Option(gloss = "Parameter \"k\" which controls the base for variable-length compression of value ranks")
	public int valueRadix = 6;

	@Option(gloss = "Fraction of hash table array actually used for entries (lower means more memory/more speed)")
	public double hashTableLoadFactor = 0.7;

	@Option(gloss = "Whether or not to store suffix indexes (necessary when using context-encoded calls to the LM")
	public boolean storeSuffixIndexes = true;

	@Option
	public double unknownWordLogProb = -100.0f;

}