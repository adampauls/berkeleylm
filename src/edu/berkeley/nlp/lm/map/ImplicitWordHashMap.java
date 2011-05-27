package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.MurmurHash;

/**
 * Low-level hash map which stored context-encoded parent pointers in a trie.
 * 
 * @author adampauls
 * 
 */
final class ImplicitWordHashMap implements Serializable, HashMap
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@PrintMemoryCount
	private final LongArray keys;

	@PrintMemoryCount
	private final long[] wordRangesLow;

	@PrintMemoryCount
	private final long[] wordRangesHigh;

	private long numFilled = 0;

	private static final int EMPTY_KEY = -1;

	public ImplicitWordHashMap(final LongArray numNgramsForEachWord, final double maxLoadFactor) {
		final long numWords = numNgramsForEachWord.size();
		wordRangesLow = new long[(int) numWords];
		wordRangesHigh = new long[(int) numWords];
		final long totalNumNgrams = setWordRanges(numNgramsForEachWord, maxLoadFactor, numWords);
		keys = LongArray.StaticMethods.newLongArray(totalNumNgrams, totalNumNgrams, totalNumNgrams);
		Logger.logss("No word key size " + totalNumNgrams);
		keys.fill(EMPTY_KEY, totalNumNgrams);
		numFilled = 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#put(long)
	 */
	@Override
	public long put(final long key) {
		int word = AbstractNgramMap.wordOf(key);
		final long hash = hash(key, word);
		if (hash < 0) return -1L;
		final long rangeStart = wordRangesLow[word];
		final long rangeEnd = wordRangesHigh[word];
		long i = keys.linearSearch(key, rangeStart, rangeEnd, hash, EMPTY_KEY, true);
		if (keys.get(i) == EMPTY_KEY) setKey(i, key);
		numFilled++;

		return i;
	}

	/**
	 * @param numNgramsForEachWord
	 * @param maxLoadFactor
	 * @param numWords
	 * @return
	 */
	private long setWordRanges(final LongArray numNgramsForEachWord, final double maxLoadFactor, final long numWords) {
		long currStart = 0;
		for (int w = 0; w < numWords; ++w) {
			wordRangesLow[w] = currStart;
			final long numNgrams = numNgramsForEachWord.get(w);
			currStart += numNgrams <= 3 ? numNgrams : Math.round(numNgrams * 1.0 / maxLoadFactor);
			wordRangesHigh[w] = currStart;

		}
		return currStart;
	}

	private void setKey(final long index, final long putKey) {
		final long contextOffset = AbstractNgramMap.contextOffsetOf(putKey);
		assert contextOffset >= 0;
		keys.set(index, contextOffset);

	}

	public final long getOffset(final long key) {
		int word = AbstractNgramMap.wordOf(key);
		final long hash = hash(key, word);
		if (hash < 0) return -1L;
		final long rangeStart = wordRangesLow[word];
		final long rangeEnd = wordRangesHigh[word];
		final long startIndex = hash;
		assert startIndex >= rangeStart;
		assert startIndex < rangeEnd;
		return keys.linearSearch(AbstractNgramMap.contextOffsetOf(key), rangeStart, rangeEnd, startIndex, EMPTY_KEY, false);
	}

	public long getCapacity() {
		return keys.size();
	}

	public double getLoadFactor() {
		return (double) numFilled / getCapacity();
	}

	private long hash(final long key, final int firstWord) {
		final long hashed = (MurmurHash.hashOneLong(key, 31));
		long hash1 = hashed;
		if (hash1 < 0) hash1 = -hash1;
		if (wordRangesLow == null) return (int) (hash1 % getCapacity());
		if (firstWord >= wordRangesLow.length) return -1;
		final long startOfRange = wordRangesLow[firstWord];
		final long numHashPositions = wordRangesHigh[firstWord] - startOfRange;
		if (numHashPositions == 0) return -1;
		hash1 = (hash1 % numHashPositions);
		return hash1 + startOfRange;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#getNextOffset(long)
	 */
	long getNextOffset(long offset) {
		return keys.get(offset);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#getWordForContext(long)
	 */
	int getWordForContext(long contextOffset) {
		int binarySearch = Arrays.binarySearch(wordRangesLow, contextOffset);
		return binarySearch >= 0 ? binarySearch : (-binarySearch - 2);
	}

	@Override
	public long getKey(long contextOffset) {
		return AbstractNgramMap.combineToKey(getWordForContext(contextOffset), getNextOffset(contextOffset));
	}

}