package edu.berkeley.nlp.lm;

import java.io.Serializable;

import edu.berkeley.nlp.lm.io.KneserNeyLmReaderCallback;
import edu.berkeley.nlp.lm.util.Annotations.Option;

/**
 * Stores some configuration options, with useful defaults.
 * 
 * @author adampauls
 * 
 */
public class ConfigOptions implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@Option(gloss = "Number of longs (8 bytes) used as a block for variable length compression")
	public int compressedBlockSize = 16;

	@Option(gloss = "Parameter \"k\" which controls the base for variable-length compression of offset deltas")
	public int offsetDeltaRadix = 6;

	@Option(gloss = "Parameter \"k\" which controls the base for variable-length compression of value ranks")
	public int valueRadix = 6;

	@Option(gloss = "Fraction of hash table array actually used for entries (lower means more memory/more speed)")
	public double hashTableLoadFactor = 1.0/1.5;

	@Option(gloss = "Probability returned when the last word of an n-gram is not in the vocabulary of the LM (this is *not* the probability of the <unk> tag)")
	public double unknownWordLogProb = -100.0f;

	@Option(gloss = "Backoff constant used for stupid backoff")
	public double stupidBackoffAlpha = 0.4;

	@Option(gloss = "Discounts used in estimating Kneser-Ney language models (one for each order)")
	public double[] kneserNeyDiscounts = KneserNeyLmReaderCallback.defaultDiscounts();

	@Option(gloss = "Minimum token counts used in estimating Kneser-Ney language models (one for each order). Note that for some internal reasons, these counts are *only* applied to the highest- and second-highest order n-grams (for example, if you estimate a 5-gram language model, only 4- and 5-grams will be thresholded.")
	public double[] kneserNeyMinCounts = KneserNeyLmReaderCallback.defaultMinCounts();
	
	@Option(gloss = "Number of bits to round floats to when reading from ARPA LM files")
	public static int roundBits = 12;

}