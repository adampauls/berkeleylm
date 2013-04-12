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
	public double hashTableLoadFactor = 1.0 / 1.5;

	@Option(gloss = "Probability returned when the last word of an n-gram is not in the vocabulary of the LM (this is *not* the probability of the <unk> tag)")
	public double unknownWordLogProb = -100.0f;

	@Option(gloss = "Backoff constant used for stupid backoff")
	public double stupidBackoffAlpha = 0.4;

	@Option(gloss = "Discounts used in estimating Kneser-Ney language models (one for each order). If null, they are calculated automatically using c1/(c1+2*c2), where cn is the number of ngrams with count n.")
	public double[] kneserNeyDiscounts = null;//KneserNeyLmReaderCallback.defaultDiscounts();

	@Option(gloss = "Minimum token counts used in estimating Kneser-Ney language models (one for each order). Note that for some internal reasons, these counts are *only* applied to the highest- and second-highest order n-grams (for example, if you estimate a 5-gram language model, only 4- and 5-grams will be thresholded. Also, any ngram orders beyond the length of this array are considered to have min count 0.")
	public double[] kneserNeyMinCounts = KneserNeyLmReaderCallback.defaultMinCounts();

	@Option(gloss = "Number of bits allocated for a word in a context encoding (remaining bits of a long are used to encode an offset")
	public int numWordBits = 26;

	@Option(gloss = "Whether to lock indexers after language model creation. This prevents the vocabulary from growing.")
	public boolean lockIndexer = true;

	@Option(gloss = "Number of bits to round the mantissa of floats to when reading from ARPA LM files. Note that the mantissa of a float is at most 24 bits long.")
	public static int roundBits = 24;

	@Option(gloss = "For (uncompressed) models that store probabilities and backoffs, store by ranking")
	public boolean storeRankedProbBackoffs = true;


	public ConfigOptions() {
	}

}
