package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.hash.HashFunction;
import edu.berkeley.nlp.lm.util.hash.MurmurHash;
import edu.berkeley.nlp.lm.values.ValueContainer;

public class HashNgramMap<T> extends AbstractNgramMap<T> implements ContextEncodedNgramMap<T>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private static final int PRIME = 31;

	private static final int NUM_INDEX_BITS = 36;

	private static final int WORD_BIT_OFFSET = NUM_INDEX_BITS;

	private static final int INDEX_OFFSET = 0;

	private static final long SUFFIX_MASK = mask(NUM_INDEX_BITS, INDEX_OFFSET);

	transient private long[] cachedLastIndex = new long[6];

	transient private int[][] cachedLastSuffix = new int[6][];

	private final boolean cacheSuffixes;

	@PrintMemoryCount
	private final HashMap[] maps;

	private final HashFunction hashFunction;

	private final double maxLoadFactor;

	private final long initialCapacity;

	private long numWords = 0;

	private final boolean useContextEncoding;

	private final LongArray[] numNgramsForEachWord;

	private final boolean reversed = false;

	public HashNgramMap(final ValueContainer<T> values, final HashFunction hashFunction, final NgramMapOpts opts, final LongArray[] numNgramsForEachWord,
		final long initialCapacity) {
		super(values, opts);
		this.numNgramsForEachWord = numNgramsForEachWord;
		this.cacheSuffixes = opts.cacheSuffixes;
		this.useContextEncoding = opts.storePrefixIndexes || opts.reverseTrie;
		maps = new HashNgramMap.HashMap[numNgramsForEachWord.length];
		for (int ngramOrder = 0; ngramOrder < numNgramsForEachWord.length; ++ngramOrder) {
			maps[ngramOrder] = new HashMap(numNgramsForEachWord[ngramOrder]);
			values.setSizeAtLeast(sum(numNgramsForEachWord[ngramOrder]), ngramOrder);
		}
		this.hashFunction = hashFunction;
		this.initialCapacity = initialCapacity;
		this.maxLoadFactor = opts.maxLoadFactor;
	}

	//	/**
	//	 * For rehashing
	//	 * 
	//	 * @param numNgramsForEachWord
	//	 */
	//	private HashNgramMap(final ValueContainer<T> values, final HashFunction hashFunction, final NgramMapOpts opts, final long[] capacities,
	//		final long[][] numNgramsForEachWord) {
	//		super(values, opts, opts.reverseTrie);
	//		this.cacheSuffixes = opts.cacheSuffixes;
	//		this.useContextEncoding = opts.storePrefixIndexes || opts.reverseTrie;
	//		maps = new HashNgramMap.HashMap[6];
	//		for (int i = 0; i < capacities.length; ++i) {
	//			if (capacities[i] < 0) continue;
	//			maps[i] = new HashMap(capacities[i], numNgramsForEachWord == null ? null : numNgramsForEachWord[i], opts.reverseTrie);
	//		}
	//		this.hashFunction = hashFunction;
	//		this.initialCapacity = -1L;
	//		this.maxLoadFactor = 0.75;
	//	}

	@Override
	public long put(final int[] ngram, final T val) {

		final HashMap tightHashMap = maps[ngram.length - 1];
		//		if (tightHashMap == null) {
		//			final long capacity = ngram.length == 1 ? initialCapacity : maps[ngram.length - 2].getCapacity();
		//			tightHashMap = maps[ngram.length - 1] = new HashMap(capacity, null, reversed);
		//		}
		return addHelp(ngram, 0, ngram.length, val, tightHashMap, true);

	}

	/**
	 * @param ngram
	 * @param val
	 * @param tightHashMap
	 */
	private long addHelp(final int[] ngram, final int startPos, final int endPos, final T val, final HashMap map, final boolean rehashIfNecessary) {

		final long key = getKey(ngram, startPos, endPos, true);
		return addHelpWithKey(ngram, startPos, endPos, val, map, key, rehashIfNecessary);
	}

	/**
	 * @param ngram
	 * @param val
	 * @param map
	 * @param hash
	 * @param key
	 * @param rehashIfNecessary
	 * @return
	 */
	private long addHelpWithKey(final int[] ngram, final int startPos, final int endPos, final T val, final HashMap map, final long key,
		final boolean rehashIfNecessary) {

		final long hash = hash(ngram, startPos, endPos, map);
		final long index = map.put(hash, key);
		long suffixIndex = -1;
		if (endPos - startPos > 1) {
			final int nextStartPos = reversed ? startPos : (startPos + 1);
			final int nextEndPos = reversed ? (endPos - 1) : (endPos);
			final HashMap suffixMap = maps[endPos - startPos - 2];
			long suffixHash = -1;
			suffixHash = hash(ngram, nextStartPos, nextEndPos, suffixMap);
			suffixIndex = suffixMap.getIndexImplicitly(ngram, suffixHash, nextStartPos, nextEndPos, maps);
			if (suffixIndex < 0) {
				suffixIndex = addHelp(ngram, nextStartPos, nextEndPos, null, suffixMap, false);
			}
			assert suffixIndex >= 0;
		}
		values.add(endPos - startPos - 1, index, prefixOffsetOf(key), wordOf(key), val, suffixIndex);
		if (rehashIfNecessary && map.getLoadFactor() > maxLoadFactor) {
			assert false;
			//			rehash(endPos - 1, map.getCapacity() * 3 / 2, false);
		}
		return index;
	}

	private long getKey(final int[] ngram, final int startPos, final int endPos, final boolean addIfNecessary) {
		long key = combineToKey(reversed ? ngram[endPos - 1] : ngram[startPos], 0);
		if (endPos - startPos == 1) return key;
		for (int ngramOrder = 1; ngramOrder < endPos - startPos; ++ngramOrder) {
			final int currNgramPos = reversed ? (endPos - ngramOrder) : (startPos + ngramOrder);
			final int currStartPos = reversed ? currNgramPos : startPos;
			final int currEndPos = reversed ? endPos : currNgramPos;

			final HashMap currMap = maps[ngramOrder - 1];
			final long hash = //useContextEncoding ? hash(key, ngram[currNgramPos - 1], ngramOrder - 1, currMap) : 
			hash(ngram, currStartPos, currEndPos, currMap);
			long index = hash < 0 ? -1L : getIndexHelp(ngram, currStartPos, ngramOrder, currEndPos, hash);
			if (index == -1L) {
				if (addIfNecessary) {
					index = addHelp(ngram, currStartPos, currEndPos, null, currMap, false);
					Arrays.fill(cachedLastSuffix, null);
				} else
					return -1;
			}

			key = combineToKey(reversed ? ngram[currNgramPos - 1] : ngram[currNgramPos], index);
		}
		return key;
	}

	private long hash(final long key, final int firstWord, final int ngramOrder, final HashMap currMap) {
		assert useContextEncoding;
		final long hashed = (MurmurHash.hashOneLong(key, 31)) + ngramOrder;
		return processHash(hashed, firstWord, currMap);
	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param ngramOrder
	 * @param currEndPos
	 * @param hash
	 * @return
	 */
	private long getIndexHelp(final int[] ngram, final int startPos, final int ngramOrder, final int endPos, final long hash) {
		if (cacheSuffixes) {
			if (cachedLastSuffix[endPos - startPos - 1] != null && equals(ngram, startPos, endPos, cachedLastSuffix[endPos - startPos - 1])) { //
				return cachedLastIndex[endPos - startPos - 1];
			}
		}
		final long index = maps[ngramOrder - 1].getIndexImplicitly(ngram, hash, startPos, endPos, maps);
		if (cacheSuffixes) {
			cachedLastSuffix[endPos - startPos - 1] = getSubArray(ngram, startPos, endPos);
			cachedLastIndex[endPos - startPos - 1] = index;
		}
		return index;
	}

	@Override
	public void handleNgramsFinished(final int justFinishedOrder) {
		final int ngramOrder = justFinishedOrder - 1;

		numWords = Math.max(numWords, maps[ngramOrder].maxWord + 1);

	}

	@Override
	public long getOffset(final int[] ngram, final int startPos, final int endPos) {
		if (containsOutOfVocab(ngram, startPos, endPos)) return -1;

		final HashMap tightHashMap = maps[endPos - startPos - 1];
		final long hash = hash(ngram, startPos, endPos, tightHashMap);
		if (hash < 0) return -1;
		final long index = tightHashMap.getIndexImplicitly(ngram, hash, startPos, endPos, maps);
		return index;
	}

	@Override
	public void trim() {

		//		if (opts.storeWordsImplicitly) {
		//			Logger.startTrack("Implicitizing words");
		//			rehash(-1, -1, true);
		//			Logger.endTrack();
		//		}
		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
			if (maps[ngramOrder] == null) break;
			values.trimAfterNgram(ngramOrder, maps[ngramOrder].getCapacity());
			Logger.logss("Load factor for " + (ngramOrder + 1) + ": " + maps[ngramOrder].getLoadFactor());

		}

	}

	@Override
	public T getValue(final int[] ngram, final int startPos, final int endPos, final LmContextInfo prefixIndex) {
		final long index = getOffset(ngram, startPos, endPos);
		return values.getFromOffset(index, endPos - startPos);
	}

	@Override
	public ValueOffsetPair<T> getValueAndOffset(final long contextOffset, final int prefixNgramOrder, final int word) {
		assert false : "Untested";
		final long offset = getOffset(contextOffset, prefixNgramOrder, word);
		return new ValueOffsetPair<T>(values.getFromOffset(offset, prefixNgramOrder + 1), offset);
	}

	/**
	 * @param ngram
	 * @param endPos
	 * @param startPos
	 * @return
	 */
	private long hash(final int[] ngram, final int startPos, final int endPos, final HashMap currMap) {
		final int firstWord = firstWord(ngram, startPos, endPos);
		if (useContextEncoding) {
			final long key = getKey(ngram, startPos, endPos, false);
			if (key < 0) return -1;
			return hash(key, firstWord, endPos - startPos - 1, currMap);
		}
		int l = (int) hashFunction.hash(ngram, startPos, endPos, PRIME);
		if (l < 0) l = -l;
		return processHash(l, firstWord, currMap);
	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @return
	 */
	private int firstWord(final int[] ngram, final int startPos, final int endPos) {
		return reversed ? ngram[startPos] : ngram[endPos - 1];
	}

	/**
	 * @param startPos
	 * @param endPos
	 * @param hash
	 * @param firstWord
	 * @return
	 */
	private long processHash(final long hash_, final int firstWord, final HashMap currMap) {
		long hash = hash_;
		if (hash < 0) hash = -hash;
		final long numHashPositions = currMap.getNumHashPositions(firstWord);
		if (numHashPositions == 0) return -1;
		hash = (int) (hash % numHashPositions);
		return hash + currMap.getStartOfRange(firstWord);
	}

	private static long mask(final int i, final int bitOffset) {
		return ((1L << i) - 1L) << bitOffset;
	}

	private static long prefixOffsetOf(final long currKey) {
		return (currKey & SUFFIX_MASK) >>> INDEX_OFFSET;
	}

	private static int wordOf(final long currKey) {
		return (int) (currKey >>> WORD_BIT_OFFSET);
	}

	private static long combineToKey(final int word, final long suffix) {
		return ((long) word << WORD_BIT_OFFSET) | (suffix << INDEX_OFFSET);
	}

	//	private void rehash(final int changedNgramOrder, final long newCapacity, final boolean storeWordsImplicitly) {
	//		final ValueContainer<T> newValues = values.createFreshValues();
	//		final long[] newCapacities = new long[maps.length];
	//		Arrays.fill(newCapacities, -1L);
	//		long[][] numNgramsForEachWord = null;
	//		if (storeWordsImplicitly) {
	//			numNgramsForEachWord = new long[maps.length][(int) numWords];
	//			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
	//				final HashMap currMap = maps[ngramOrder];
	//				if (currMap == null) continue;
	//				for (long actualIndex = 0; actualIndex < currMap.getCapacity(); ++actualIndex) {
	//					final long key = currMap.getKey(actualIndex);
	//					if (key == HashMap.EMPTY_KEY) continue;
	//					//					final int[] ngram = getNgram(key, ngramOrder);
	//					final int firstWordOfNgram = wordOf(key);//opts.reverseTrie ? ngram[0] : ngram[ngramOrder];
	//					numNgramsForEachWord[ngramOrder][firstWordOfNgram]++;
	//				}
	//
	//				for (int i = 0; i < numNgramsForEachWord[ngramOrder].length; ++i) {
	//					final long numNgrams = numNgramsForEachWord[ngramOrder][i];
	//					numNgramsForEachWord[ngramOrder][i] = numNgrams <= 3 ? numNgrams : Math.round(numNgrams * 1.0 / maxLoadFactor);
	//				}
	//			}
	//			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
	//				if (maps[ngramOrder] == null) break;
	//				newCapacities[ngramOrder] = sum(numNgramsForEachWord[ngramOrder]);
	//			}
	//		} else {
	//
	//			for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
	//				if (maps[ngramOrder] == null) break;
	//				newCapacities[ngramOrder] = ngramOrder == changedNgramOrder ? newCapacity : maps[ngramOrder].getCapacity();
	//			}
	//		}
	//
	//		final HashNgramMap<T> newMap = new HashNgramMap<T>(newValues, hashFunction, opts, newCapacities, numNgramsForEachWord);
	//
	//		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
	//			final HashMap currMap = maps[ngramOrder];
	//			if (currMap == null) continue;
	//			for (long actualIndex = 0; actualIndex < currMap.getCapacity(); ++actualIndex) {
	//				final long key = currMap.getKey(actualIndex);
	//				if (key == HashMap.EMPTY_KEY) continue;
	//				final int[] ngram = getNgram(key, ngramOrder);
	//
	//				final T val = values.getFromOffset(actualIndex, ngramOrder);
	//
	//				newMap.addHelp(ngram, 0, ngram.length, val, newMap.maps[ngramOrder], false);
	//
	//			}
	//		}
	//		maps = newMap.maps;
	//
	//		values.setFromOtherValues(newValues);
	//
	//	}

	private static long sum(final long[] array) {
		long sum = 0;
		for (final long l : array)
			sum += l;
		return sum;
	}

	private static long sum(final LongArray array) {
		long sum = 0;
		for (long i = 0; i < array.size(); ++i)
			sum += array.get(i);
		return sum;
	}

	private int[] getNgram(final long key_, final int ngramOrder_) {
		long key = key_;
		int ngramOrder = ngramOrder_;
		final int[] l = new int[ngramOrder + 1];
		int firstWord = wordOf(key);
		int k = reversed ? 0 : (l.length - 1);
		l[k] = firstWord;

		if (reversed)
			k++;
		else
			k--;
		while (ngramOrder > 0) {
			final long suffixIndex = prefixOffsetOf(key);
			key = maps[ngramOrder - 1].getKey(suffixIndex);
			ngramOrder--;
			firstWord = wordOf(key);
			l[k] = firstWord;
			if (reversed)
				k++;
			else
				k--;
		}
		return l;
	}

	@Override
	public void initWithLengths(final List<Long> numNGrams) {
		//		maps = new HashMap[numNGrams.size()];
		//		for (int i = 0; i < numNGrams.size(); ++i) {
		//			final long l = numNGrams.get(i);
		//			final long size = Math.round(l / maxLoadFactor) + 1;
		//			maps[i] = new HashMap(size, null, false);
		//			values.setSizeAtLeast(size, i);
		//
		//		}
	}

	@Override
	public long getOffset(final long contextOffset_, final int contextOrder, final int word) {
		final long contextOffset = contextOrder < 0 ? 0 : contextOffset_;
		assert contextOffset >= 0;
		final int ngramOrder = contextOrder + 1;

		final long key = combineToKey(word, contextOffset);
		final HashMap map = maps[ngramOrder];
		final long hash = hash(key, word, ngramOrder, map);
		if (hash < 0) return -1L;
		final long index = map.getIndexImplicity(contextOffset, word, hash);
		return index;
	}

	private static final class HashMap implements Serializable
	{

		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		@PrintMemoryCount
		final LongArray keys;

		@PrintMemoryCount
		final long[] wordRangesLow;

		@PrintMemoryCount
		final long[] wordRangesHigh;

		long numFilled = 0;

		long maxWord = 0;

		private final boolean reverseTrie;

		private static final int EMPTY_KEY = -1;

		public HashMap(final LongArray numNgramsForEachWord) {
			final long numWords = numNgramsForEachWord.size();
			wordRangesLow = new long[(int) numWords];
			wordRangesHigh = new long[(int) numWords];
			long currStart = 0;
			for (int w = 0; w < numWords; ++w) {
				wordRangesLow[w] = currStart;
				currStart += numNgramsForEachWord.get(w);
				wordRangesHigh[w] = currStart;

			}
			keys = LongArray.StaticMethods.newLongArray(currStart, currStart, currStart);
			Logger.logss("No word key size " + currStart);
			keys.fill(EMPTY_KEY, currStart);
			reverseTrie = false;
			numFilled = 0;
		}

		private long getKey(final long index) {
			return keys.get(index);
		}

		private final long getNext(final long i_, final long start, final long end) {
			long i = i_;
			++i;
			if (i >= end) i = start;
			return i;
		}

		public long put(final long index, final long putKey) {
			final int firstWordOfNgram = wordOf(putKey);
			final long rangeStart = wordRangesLow[firstWordOfNgram];
			final long rangeEnd = wordRangesHigh[firstWordOfNgram];
			long searchKey = getKey(index);
			long i = index;
			while (searchKey != EMPTY_KEY && searchKey != putKey) {

				i = getNext(i, rangeStart, rangeEnd);

				searchKey = getKey(i);
			}

			if (searchKey == EMPTY_KEY) setKey(i, putKey);
			numFilled++;
			maxWord = Math.max(maxWord, firstWordOfNgram);

			return i;
		}

		private void setKey(final long index, final long putKey) {

			assert keys.get(index) == EMPTY_KEY;
			final long contextOffset = prefixOffsetOf(putKey);
			assert contextOffset >= 0;
			keys.set(index, contextOffset);

		}

		//		public final long getIndex(final int[] ngram, final long index, final int startPos, final int endPos, final HashMap[] maps) {
		//			if (keys == null) return getIndexImplicity(ngram, index, startPos, endPos, maps);
		//			final LongArray localKeys = keys;
		//			final int firstWordOfNgram = reverseTrie ? ngram[startPos] : ngram[endPos - 1];
		//			final int keysLength = keys.length;
		//			final int rangeStart = 0;
		//			final int rangeEnd = keysLength;
		//			assert index >= rangeStart;
		//			assert index < rangeEnd;
		//			int i = (int) index;
		//			int num = 1;
		//			while (true) {
		//				final long searchKey = localKeys[i];
		//				final int next = getNext(i, rangeStart, rangeEnd, num);
		//				num++;
		//				if (searchKey == EMPTY_KEY) {//
		//					return -1L;
		//				}
		//				if (firstWordOfNgram == wordOf(searchKey) && suffixEquals(prefixOffsetOf(searchKey), ngram, startPos, endPos, maps, reverseTrie)) { //
		//					return i;
		//				}
		//				i = next;
		//
		//			}
		//		}
		//
		//		public final long getIndex(final long key, final long suffixIndex, final int firstWord, final long index) {
		//			if (keys == null) return getIndexImplicity(suffixIndex, firstWord, index);
		//			final long[] localKeys = keys;
		//			final int keysLength = keys.length;
		//			final int rangeStart = 0;
		//			final int rangeEnd = keysLength;
		//			assert index >= rangeStart;
		//			assert index < rangeEnd;
		//			int i = (int) index;
		//			while (true) {
		//				final long searchKey = localKeys[i];
		//				if (searchKey == key) { //
		//					return i;
		//				}
		//				if (searchKey == EMPTY_KEY) {//
		//					return -1L;
		//				}
		//				++i;
		//				if (i >= rangeEnd) i = rangeStart;
		//
		//			}
		//		}

		public final long getIndexImplicity(final long contextOffset, final int word, final long startIndex) {
			final LongArray localKeys = keys;
			final long rangeStart = wordRangesLow[word];
			final long rangeEnd = wordRangesHigh[word];
			assert startIndex >= rangeStart;
			assert startIndex < rangeEnd;
			long i = startIndex;
			boolean goneAroundOnce = false;
			while (true) {
				if (i == rangeEnd) {
					if (goneAroundOnce) return -1L;
					i = rangeStart;
					goneAroundOnce = true;
				}
				final long searchKey = localKeys.get(i);
				if (searchKey == contextOffset) {//
					return i;
				}
				if (searchKey == EMPTY_KEY) {//
					return -1L;
				}
				++i;

			}
		}

		public final long getIndexImplicitly(final int[] ngram, final long index, final int startPos, final int endPos, final HashMap[] maps) {
			final LongArray localKeys = keys;
			final int firstWordOfNgram = reverseTrie ? ngram[startPos] : ngram[endPos - 1];
			final long rangeStart = wordRangesLow[firstWordOfNgram];
			final long rangeEnd = wordRangesHigh[firstWordOfNgram];
			assert index >= rangeStart;
			assert index < rangeEnd;
			long i = index;
			boolean goneAroundOnce = false;
			while (true) {
				if (i == rangeEnd) {
					if (goneAroundOnce) return -1L;
					i = rangeStart;
					goneAroundOnce = true;
				}
				final long searchKey = localKeys.get(i);
				if (searchKey == EMPTY_KEY) {//
					return -1L;
				}
				if (implicitSuffixEquals(searchKey, ngram, startPos, endPos, maps, reverseTrie)) { //
					return i;
				}
				++i;

			}
		}

		public long getCapacity() {
			return keys.size();
		}

		public double getLoadFactor() {
			return (double) numFilled / getCapacity();
		}

		private static final boolean suffixEquals(final long suffixIndex_, final int[] ngram, final int startPos, final int endPos, final HashMap[] localMaps,
			final boolean reverse) {
			return reverse ? suffixEqualsReverse(suffixIndex_, ngram, startPos + 1, endPos, localMaps) : suffixEqualsForward(suffixIndex_, ngram, startPos,
				endPos - 1, localMaps);

		}

		private static final boolean suffixEqualsForward(final long suffixIndex_, final int[] ngram, final int startPos, final int endPos,
			final HashMap[] localMaps) {
			long suffixIndex = suffixIndex_;
			for (int pos = endPos - 1; pos >= startPos; --pos) {
				final HashMap suffixMap = localMaps[pos - startPos];
				final long currKey = suffixMap.getKey(suffixIndex);
				final int firstWord = wordOf(currKey);
				if (ngram[pos] != firstWord) return false;
				if (pos == startPos) return true;
				suffixIndex = prefixOffsetOf(currKey);
			}
			return true;

		}

		private static final boolean suffixEqualsReverse(final long suffixIndex_, final int[] ngram, final int startPos, final int endPos,
			final HashMap[] localMaps) {
			long suffixIndex = suffixIndex_;
			for (int pos = startPos; pos < endPos; ++pos) {
				final HashMap suffixMap = localMaps[endPos - pos - 1];
				final long currKey = suffixMap.getKey(suffixIndex);
				final int firstWord = wordOf(currKey);
				if (ngram[pos] != firstWord) return false;
				if (pos == endPos - 1) return true;
				suffixIndex = prefixOffsetOf(currKey);
			}
			return true;

		}

		private static final boolean implicitSuffixEquals(final long contextOffset_, final int[] ngram, final int startPos, final int endPos,
			final HashMap[] localMaps, final boolean reverse) {
			return reverse ? implicitSuffixEqualsReverse(contextOffset_, ngram, startPos + 1, endPos, localMaps) : implicitSuffixEqualsForward(contextOffset_,
				ngram, startPos, endPos - 1, localMaps);
		}

		private static final boolean implicitSuffixEqualsForward(final long contextOffset_, final int[] ngram, final int startPos, final int endPos,
			final HashMap[] localMaps) {
			long contextOffset = contextOffset_;
			for (int pos = endPos - 1; pos >= startPos; --pos) {
				final HashMap suffixMap = localMaps[pos - startPos];
				final int firstSearchWord = ngram[pos];
				if (firstSearchWord >= suffixMap.wordRangesLow.length) return false;
				final long rangeStart = suffixMap.wordRangesLow[firstSearchWord];
				if (contextOffset < rangeStart) return false;
				final long rangeEnd = suffixMap.wordRangesHigh[firstSearchWord];
				if (contextOffset >= rangeEnd) return false;
				//				final int firstWord = firstWord(currKey);
				if (pos == startPos) return true;
				final long currKey = suffixMap.getKey(contextOffset);
				contextOffset = prefixOffsetOf(currKey);
			}
			return true;

		}

		private static final boolean implicitSuffixEqualsReverse(final long contextOffset_, final int[] ngram, final int startPos, final int endPos,
			final HashMap[] localMaps) {
			long contextOffset = contextOffset_;
			for (int pos = startPos; pos < endPos; ++pos) {
				final HashMap suffixMap = localMaps[endPos - pos - 1];
				//				final int firstWord = firstWord(currKey);
				final int firstSearchWord = ngram[pos];
				if (firstSearchWord >= suffixMap.wordRangesLow.length) return false;
				final long rangeStart = suffixMap.wordRangesLow[firstSearchWord];
				if (contextOffset < rangeStart) return false;
				final long rangeEnd = suffixMap.wordRangesHigh[firstSearchWord];
				if (contextOffset >= rangeEnd) return false;
				if (pos == endPos - 1) return true;
				final long currKey = suffixMap.getKey(contextOffset);
				contextOffset = prefixOffsetOf(currKey);
			}
			return true;

		}

		public long getNumHashPositions(final int word) {
			if (wordRangesLow == null) return getCapacity();
			if (word >= wordRangesLow.length) return 0;
			return wordRangesHigh[word] - wordRangesLow[word];
		}

		public long getStartOfRange(final int word) {
			if (wordRangesLow == null) return 0;
			return wordRangesLow[word];
		}
	}

}
