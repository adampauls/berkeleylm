package edu.berkeley.nlp.lm.map;

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

	private long numWords = 0;

	private final boolean useContextEncoding;

	private final boolean storeSuffixOffsets;

	private final boolean reversed = false;

	public HashNgramMap(final ValueContainer<T> values, final HashFunction hashFunction, final NgramMapOpts opts, final LongArray[] numNgramsForEachWord) {
		super(values, opts);
		this.cacheSuffixes = opts.cacheSuffixes;
		this.useContextEncoding = opts.storePrefixIndexes || opts.reverseTrie;
		this.storeSuffixOffsets = opts.storePrefixIndexes;
		this.hashFunction = hashFunction;
		maps = new HashMap[numNgramsForEachWord.length];
		this.maxLoadFactor = opts.maxLoadFactor;
		for (int ngramOrder = 0; ngramOrder < numNgramsForEachWord.length; ++ngramOrder) {
			maps[ngramOrder] = new HashMap(numNgramsForEachWord[ngramOrder], maxLoadFactor);
			values.setSizeAtLeast(maps[ngramOrder].getCapacity(), ngramOrder);
		}
	}

	@Override
	public long put(final int[] ngram, final T val) {
		final int endPos = ngram.length;
		final HashMap map = maps[ngram.length - 1];
		final long key = getKey(ngram, 0, endPos);
		final long hash = hash(ngram, 0, endPos, map);
		final long index = map.put(hash, key);
		final long suffixIndex = getSuffixOffset(ngram, 0, endPos);
		values.add(endPos - 0 - 1, index, contextOffsetOf(key), wordOf(key), val, suffixIndex);
		return index;
	}

	/**
	 * @param ngram
	 * @param endPos
	 * @return
	 */
	private long getSuffixOffset(final int[] ngram, final int startPos, final int endPos) {
		long suffixIndex = -1;
		if (storeSuffixOffsets && endPos - startPos > 1) {
			final int start = reversed ? startPos : (startPos + 1);
			final int end = reversed ? (endPos - 1) : (endPos);
			final HashMap suffixMap = maps[end - start - 1];
			long suffixHash = -1;
			suffixHash = hash(ngram, start, end, suffixMap);
			if (suffixHash >= 0) {
				suffixIndex = suffixMap.getIndexImplicitly(ngram, suffixHash, start, end, maps);
			}
		}
		return suffixIndex;
	}

	private long getKey(final int[] ngram, final int startPos, final int endPos) {
		long key = combineToKey(reversed ? ngram[endPos - 1] : ngram[startPos], 0);
		if (endPos - startPos == 1) return key;
		for (int ngramOrder = 1; ngramOrder < endPos - startPos; ++ngramOrder) {
			final int currNgramPos = reversed ? (endPos - ngramOrder) : (startPos + ngramOrder);
			final int currStartPos = reversed ? currNgramPos : startPos;
			final int currEndPos = reversed ? endPos : currNgramPos;

			final HashMap currMap = maps[ngramOrder - 1];
			final long hash = hash(ngram, currStartPos, currEndPos, currMap);
			final long index = hash < 0 ? -1L : getIndexHelp(ngram, currStartPos, ngramOrder, currEndPos, hash);
			if (index == -1L) { return -1; }

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
	public void initWithLengths(final List<Long> numNGrams) {
	}

	@Override
	public long getOffset(final long contextOffset_, final int contextOrder, final int word) {
		final long contextOffset = Math.max(contextOffset_, 0);
		final int ngramOrder = contextOrder + 1;

		final long key = combineToKey(word, contextOffset);
		final HashMap map = maps[ngramOrder];
		final long hash = hash(key, word, ngramOrder, map);
		if (hash < 0) return -1L;
		final long index = map.getIndexImplicity(contextOffset, word, hash);
		return index;
	}

	@Override
	public LmContextInfo getOffsetForNgram(final int[] ngram, final int startPos, final int endPos) {
		final LmContextInfo lmContextInfo = new LmContextInfo();
		for (int start = endPos - 1; start >= startPos; --start) {
			final long offset = getOffset(ngram, start, endPos);
			if (offset < 0) break;
			lmContextInfo.offset = offset;
			lmContextInfo.order = endPos - start - 1;

		}
		return lmContextInfo;
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
			final long key = getKey(ngram, startPos, endPos);
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

	static long contextOffsetOf(final long currKey) {
		return (currKey & SUFFIX_MASK) >>> INDEX_OFFSET;
	}

	static int wordOf(final long currKey) {
		return (int) (currKey >>> WORD_BIT_OFFSET);
	}

	private static long combineToKey(final int word, final long suffix) {
		return ((long) word << WORD_BIT_OFFSET) | (suffix << INDEX_OFFSET);
	}

}
