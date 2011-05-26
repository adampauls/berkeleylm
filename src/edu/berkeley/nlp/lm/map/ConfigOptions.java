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

	@Option
	public double unknownWordLogProb = -100.0f;

	@Option
	public double stupidBackoffAlpha = 0.4;

	@Option(gloss = "Number of threads to use while reading directories in the format used by Google N-grams. 0 means no threading is used")
	public int numGoogleLoadThreads = 0;

	@Option(gloss = "Whether to use compression. This should reduce memory usage significantly, but also incur a performance penalty. Context-encoded lookups are not supported with compression.")
	public boolean compress = false;

}