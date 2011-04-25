package edu.berkeley.nlp.lm.map;

import edu.berkeley.nlp.lm.util.Annotations.Option;

public class NgramMapOpts
{

	/**
	 * Number of longs (8 bytes) used as a "block" for variable length
	 * compression.
	 */
	@Option
	public int compressedBlockSize = 16;

	/**
	 * Use variable-length compression
	 */
	@Option
	public boolean compress = false;

	/**
	 * 
	 */
	@Option
	public int suffixRadix = 6;

	@Option
	public boolean buildIndex = true;

	@Option
	public boolean directBinarySearch = false;

	@Option
	public boolean useHash = false;

	@Option
	public boolean timersOn = false;

	@Option
	public boolean interpolationSearch = false;

	@Option
	public boolean averageInterpolate = false;

	@Option
	public boolean cacheSuffixes = false;

	@Option
	public boolean storeWordsImplicitly = false;

	@Option
	public double maxLoadFactor = 0.7;

	@Option
	public boolean countDeltas = false;

	@Option
	public int numGoogleLoadThreads = 0;

	@Option
	public boolean skipCompressingVals;

	@Option
	public boolean absDeltas = false;

	@Option
	public int valueRadix = 6;

	@Option
	public boolean useHuffman = false;

	@Option
	public int miniIndexNum = -1;

	@Option
	public boolean backEndCache = false;

	@Option
	public int huffmanCountCutoff = 1 << 16;

	@Option
	public boolean skipLinearSearch = false;

	@Option
	public boolean storePrefixIndexes = false;

	@Option
	public boolean logJoshuaLmRequests = false;

	@Option
	public boolean quadraticProbing = false;

	@Option
	public double unknownWordLogProb = -100.0f;

	@Option
	public boolean eliasFano;

	@Option
	public boolean reverseTrie = false;

}