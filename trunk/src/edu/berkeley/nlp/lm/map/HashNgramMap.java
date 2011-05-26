package edu.berkeley.nlp.lm.map;

import java.util.List;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.hash.MurmurHash;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ValueContainer;

public class HashNgramMap<T> extends AbstractNgramMap<T> implements ContextEncodedNgramMap<T>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@PrintMemoryCount
	private final HashMap[] maps;

	private final double maxLoadFactor;

	private long numWords = 0;

	private final boolean reversed;

	public HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final LongArray[] numNgramsForEachWord, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		maps = new HashMap[numNgramsForEachWord.length];
		for (int ngramOrder = 0; ngramOrder < numNgramsForEachWord.length; ++ngramOrder) {
			maps[ngramOrder] = new HashMap(numNgramsForEachWord[ngramOrder], maxLoadFactor, reversed);
			values.setSizeAtLeast(maps[ngramOrder].getCapacity(), ngramOrder);
		}
	}

	@Override
	public long put(final int[] ngram, final T val) {
		final int endPos = ngram.length;
		final HashMap map = maps[ngram.length - 1];
		final long key = getKey(ngram, 0, endPos);
		if (key < 0) return -1L;
		final long hash = hash(ngram, 0, endPos, map);
		final long index = map.put(hash, key);
		final long suffixIndex = getSuffixOffset(ngram, 0, endPos);
		values.add(endPos - 0 - 1, index, contextOffsetOf(key), wordOf(key), val, suffixIndex);
		return index;
	}

	//	@Override
	//	public void getValue(final int[] ngram, final int startPos, final int endPos, final @OutputParameter LmContextInfo contextOutput,
	//		final @OutputParameter T outputVal) {
	//		final long index = getOffset(ngram, startPos, endPos);
	//		values.getFromOffset(index, endPos - startPos, outputVal);
	//	}

	@Override
	public long getValueAndOffset(long contextOffset, int contextOrder, int word, @OutputParameter T outputVal) {
		return getOffsetHelp(contextOffset, contextOrder, word, outputVal);
	}

	@Override
	public long getOffset(final long contextOffset, final int contextOrder, final int word) {
		return getOffsetHelp(contextOffset, contextOrder, word, null);
	}

	/**
	 * @param contextOffset_
	 * @param contextOrder
	 * @param word
	 * @return
	 */
	private long getOffsetHelp(final long contextOffset_, final int contextOrder, final int word, @OutputParameter T outputVal) {
		final long contextOffset = Math.max(contextOffset_, 0);
		final int ngramOrder = contextOrder + 1;

		final long key = getKey(word, contextOffset);
		final HashMap map = maps[ngramOrder];
		final long hash = hash(key, word, ngramOrder, map);
		if (hash < 0) return -1L;
		final long offset = map.getIndexImplicity(contextOffset, word, hash);
		if (offset >= 0 && outputVal != null) {
			values.getFromOffset(offset, ngramOrder, outputVal);
		}
		return offset;
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

	@Override
	public void handleNgramsFinished(final int justFinishedOrder) {
		final int ngramOrder = justFinishedOrder - 1;
		numWords = Math.max(numWords, maps[ngramOrder].maxWord + 1);
	}

	@Override
	public void initWithLengths(final List<Long> numNGrams) {
	}

	@Override
	public void trim() {
		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
			if (maps[ngramOrder] == null) break;
			values.trimAfterNgram(ngramOrder, maps[ngramOrder].getCapacity());
			Logger.logss("Load factor for " + (ngramOrder + 1) + ": " + maps[ngramOrder].getLoadFactor());
		}
	}

	/**
	 * @param ngram
	 * @param endPos
	 * @return
	 */
	private long getSuffixOffset(final int[] ngram, final int startPos, final int endPos) {
		long suffixIndex = -1;
		if (endPos - startPos > 1) {
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
		long key = getKey(reversed ? ngram[endPos - 1] : ngram[startPos], 0);
		if (endPos - startPos == 1) return key;
		for (int ngramOrder = 1; ngramOrder < endPos - startPos; ++ngramOrder) {
			final int currNgramPos = reversed ? (endPos - ngramOrder) : (startPos + ngramOrder);
			final int currStartPos = reversed ? currNgramPos : startPos;
			final int currEndPos = reversed ? endPos : currNgramPos;

			final HashMap currMap = maps[ngramOrder - 1];
			final long hash = hash(ngram, currStartPos, currEndPos, currMap);
			final long index = hash < 0 ? -1L : getIndexHelp(ngram, currStartPos, ngramOrder, currEndPos, hash);
			if (index == -1L) { return -1; }

			key = getKey(reversed ? ngram[currNgramPos - 1] : ngram[currNgramPos], index);
		}
		return key;
	}

	private long hash(final long key, final int firstWord, final int ngramOrder, final HashMap currMap) {
		final long hashed = (MurmurHash.hashOneLong(key, 31)) + ngramOrder;
		return currMap.processHash(hashed, firstWord);
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
		final long index = maps[ngramOrder - 1].getIndexImplicitly(ngram, hash, startPos, endPos, maps);
		return index;
	}

	/**
	 * @param ngram
	 * @param endPos
	 * @param startPos
	 * @return
	 */
	private long hash(final int[] ngram, final int startPos, final int endPos, final HashMap currMap) {
		final int firstWord = firstWord(ngram, startPos, endPos);
		final long key = getKey(ngram, startPos, endPos);
		if (key < 0) return -1;
		return hash(key, firstWord, endPos - startPos - 1, currMap);

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

}
