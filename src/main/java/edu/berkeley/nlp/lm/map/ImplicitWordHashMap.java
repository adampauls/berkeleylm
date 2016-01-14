package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Iterator;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitUtils;
import edu.berkeley.nlp.lm.collections.Iterators;
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
	final CustomWidthArray keys;

	@PrintMemoryCount
	private final long[] wordRanges;

	private final HashNgramMap<?> ngramMap;

	private long numFilled = 0;

	private static final int EMPTY_KEY = 0;

	private final int numWords;

	private final int ngramOrder;

	@SuppressWarnings("unused")
	private final int totalNumWords;

	private final int maxNgramOrder;

	private final boolean fitsInInt;

	private final int numSuffixBits;

	public ImplicitWordHashMap(final LongArray numNgramsForEachWord, final long[] wordRanges, final int ngramOrder, final int maxNgramOrder,
		final long numNgramsForPreviousOrder, final int totalNumWords, final HashNgramMap<?> ngramMap, final boolean fitsInInt, final boolean storeWords) {
		this.ngramOrder = ngramOrder;
		this.ngramMap = ngramMap;
		assert ngramOrder >= 1;
		this.maxNgramOrder = maxNgramOrder;
		this.totalNumWords = totalNumWords;
		this.numWords = (int) numNgramsForEachWord.size();
		this.fitsInInt = fitsInInt;

		this.wordRanges = storeWords ? null : wordRanges;
		final long totalNumNgrams = setWordRanges(numNgramsForEachWord, numWords);
		numSuffixBits = CustomWidthArray.numBitsNeeded(numNgramsForPreviousOrder + 1);
		final int numBitsHere = numSuffixBits + (storeWords ? CustomWidthArray.numBitsNeeded(totalNumWords) : 0);
		keys = new CustomWidthArray(totalNumNgrams, numBitsHere, numBitsHere + ngramMap.getValues().numValueBits(ngramOrder));
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
		final long i = linearSearch(key, true);
		if (keys.get(i) == EMPTY_KEY) numFilled++;
		setKey(i, key);

		return i;
	}

	/**
	 * @param numNgramsForEachWord
	 * @param maxLoadFactor
	 * @param numWords
	 * @return
	 */
	private long setWordRanges(final LongArray numNgramsForEachWord, final long numWords) {
		long currStart = 0;
		for (int w = (0); w < numWords; ++w) {
			if (wordRanges != null) {
				setWordRangeStart(w, currStart);
				currStart += ngramMap.getRangeSizeForWord(numNgramsForEachWord, w);
			} else {
				currStart += numNgramsForEachWord.get(w);
			}
		}
		return wordRanges == null ? Math.round(currStart * 1.0 / ngramMap.getLoadFactor()) : currStart;
	}

	private void setKey(final long index, final long putKey) {
		final long contextOffset = wordRanges == null ? shrinkKey(putKey) : ngramMap.contextOffsetOf(putKey);
		assert contextOffset >= 0;
		keys.set(index, contextOffset + 1);

	}

	/**
	 * @param word
	 * @param suffixIndex
	 * @return
	 */
	private final long shrinkKey(final long key) {
		final int word = ngramMap.wordOf(key);
		final long suffixIndex = ngramMap.contextOffsetOf(key);
		return (((long) word) << (numSuffixBits)) | suffixIndex;
	}

	private final long expandKey(final long key) {
		final int word = (int) (key >>> numSuffixBits);
		final long suffixIndex = key & ((1L << numSuffixBits) - 1);
		return ngramMap.combineToKey(word, suffixIndex);
	}

	@Override
	public final long getOffset(final long key) {

		return linearSearch(key, false);
	}

	/**
	 * @param key
	 * @param returnFirstEmptyIndex
	 * @return
	 */
	private long linearSearch(final long key, final boolean returnFirstEmptyIndex) {
		final int word = ngramMap.wordOf(key);
		if (word >= numWords) return -1;
		final long rangeStart = wordRangeStart(word);
		final long rangeEnd = wordRangeEnd(word);
		final long numHashPositions = rangeEnd - rangeStart;
		if (numHashPositions == 0) return -1L;
		final long startIndex = hash(key, numHashPositions, rangeStart);
		final long contextOffsetOf = wordRanges == null ? shrinkKey(key) : ngramMap.contextOffsetOf(key);
		assert contextOffsetOf >= 0;
		assert word >= 0;
		assert startIndex >= rangeStart;
		assert startIndex < rangeEnd;

		final long index = keys.linearSearch(contextOffsetOf + 1, rangeStart, rangeEnd, startIndex, EMPTY_KEY, returnFirstEmptyIndex);
		return index;
	}

	@Override
	public long getCapacity() {
		return keys.size();
	}

	@Override
	public double getLoadFactor() {
		return (double) numFilled / getCapacity();
	}

	private long hash(final long key, final long numHashPositions, final long startOfRange) {

		long hash = BitUtils.abs(MurmurHash.hashOneLong(key, 0x9747b28c));
		hash %= numHashPositions;
		return hash + startOfRange;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#getNextOffset(long)
	 */
	long getNextOffset(final long offset) {
		return keys.get(offset) - 1;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.map.HashMap#getWordForContext(long)
	 */
	int getWordForContext(final long contextOffset) {
		int binarySearch = binarySearch(contextOffset);
		binarySearch = binarySearch >= 0 ? binarySearch : (-binarySearch - 2);
		while (binarySearch < numWords - 1 && wordRangeStart(binarySearch) == wordRangeEnd(binarySearch))
			binarySearch++;
		return binarySearch;
	}

	private int binarySearch(final long key) {
		int low = 0;
		int high = numWords - 1;

		while (low <= high) {
			final int mid = (low + high) >>> 1;
			final long midVal = wordRangeStart(mid);
			if (midVal < key)
				low = mid + 1;
			else if (midVal > key)
				high = mid - 1;
			else
				return mid; // key found
		}
		return -(low + 1); // key not found.
	}

	@Override
	public long getKey(final long contextOffset) {
		return wordRanges == null ? expandKey(getNextOffset(contextOffset)) : ngramMap.combineToKey(getWordForContext(contextOffset),
			getNextOffset(contextOffset));
	}

	@Override
	public boolean isEmptyKey(final long key) {
		return key == EMPTY_KEY;
	}

	@Override
	public long size() {
		return numFilled;
	}

	@Override
	public Iterable<Long> keys() {
		return Iterators.able(new KeyIterator(keys));
	}

	public static class KeyIterator implements Iterator<Long>
	{
		private final CustomWidthArray keys;

		public KeyIterator(final CustomWidthArray keys) {
			this.keys = keys;
			end = keys.size();
			next = -1;
			nextIndex();
		}

		@Override
		public boolean hasNext() {
			return end > 0 && next < end;
		}

		@Override
		public Long next() {
			final long nextIndex = nextIndex();
			return nextIndex;
		}

		long nextIndex() {
			final long curr = next;
			do {
				next++;
			} while (next < end && keys != null && keys.get(next) == EMPTY_KEY);
			return curr;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

		private long next;

		private final long end;
	}

	@Override
	public boolean hasContexts(final int word) {
		if (word >= numWords) return false;
		final long rangeStart = wordRangeStart(word);
		final long rangeEnd = wordRangeEnd(word);
		return (rangeEnd - rangeStart > 0);
	}

	private final long wordRangeStart(final int w) {
		return wordRanges == null ? 0 : wordRangeAt(w * maxNgramOrder + ngramOrder - 1);
	}

	private final long wordRangeEnd(final int w) {
		return wordRanges == null || w == numWords - 1 ? getCapacity() : wordRangeAt((w + 1) * maxNgramOrder + ngramOrder - 1);

	}

	/**
	 * @param logicalIndex
	 * @return
	 */
	private long wordRangeAt(final int logicalIndex) {
		if (fitsInInt) {
			return logicalIndex % 2 == 0 ? BitUtils.getLowInt(wordRanges[logicalIndex / 2]) : BitUtils.getHighInt(wordRanges[logicalIndex >> 1]);
		} else {
			return wordRanges[logicalIndex];
		}
	}

	private void setWordRangeStart(int w, long currStart) {
		final int logicalIndex = w * maxNgramOrder + ngramOrder - 1;
		if (fitsInInt) {
			if (logicalIndex % 2 == 0)
				wordRanges[logicalIndex / 2] = BitUtils.setLowInt(wordRanges[logicalIndex / 2], (int) currStart);
			else
				wordRanges[logicalIndex / 2] = BitUtils.setHighInt(wordRanges[logicalIndex / 2], (int) currStart);
		} else {
			wordRanges[logicalIndex] = currStart;
		}
	}

}
