package edu.berkeley.nlp.lm.map;

import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.MurmurHash;
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
			maps[ngramOrder] = new HashMap(numNgramsForEachWord[ngramOrder], maxLoadFactor);
			values.setSizeAtLeast(maps[ngramOrder].getCapacity(), ngramOrder);
		}
	}

	@Override
	public long put(final int[] ngram, final T val) {
		final int endPos = ngram.length;
		final HashMap map = maps[ngram.length - 1];
		final long key = getKey(ngram, 0, endPos);
		if (key < 0) return -1L;
		final long index = map.put(key);
		final long suffixIndex = getSuffixOffset(ngram, 0, endPos);
		values.add(endPos - 0 - 1, index, contextOffsetOf(key), wordOf(key), val, suffixIndex);
		return index;
	}

	@Override
	public long getValueAndOffset(final long contextOffset, final int contextOrder, final int word, @OutputParameter final T outputVal) {
		return getOffsetForContextEncoding(contextOffset, contextOrder, word, outputVal);
	}

	@Override
	public long getOffset(final long contextOffset, final int contextOrder, final int word) {
		return getOffsetForContextEncoding(contextOffset, contextOrder, word, null);
	}

	/**
	 * @param contextOffset_
	 * @param contextOrder
	 * @param word
	 * @return
	 */
	private long getOffsetForContextEncoding(final long contextOffset_, final int contextOrder, final int word, @OutputParameter final T outputVal) {
		final long contextOffset = Math.max(contextOffset_, 0);
		final int ngramOrder = contextOrder + 1;

		final long key = combineToKey(word, contextOffset);
		final HashMap map = maps[ngramOrder];
		//		final long hash = hash(key, word, ngramOrder, map);
		//		if (hash < 0) return -1L;
		final long offset = map.getIndexImplicity(key);
		if (offset >= 0 && outputVal != null) {
			values.getFromOffset(offset, ngramOrder, outputVal);
		}
		return offset;
	}

	@Override
	public long getOffset(final int[] ngram, final int startPos, final int endPos) {
		return getOffsetFromRawNgram(ngram, startPos, endPos);
	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @return
	 */
	private long getOffsetFromRawNgram(final int[] ngram, final int startPos, final int endPos) {
		if (containsOutOfVocab(ngram, startPos, endPos)) return -1;
		final int ngramOrder = endPos - startPos - 1;
		final HashMap currMap = maps[ngramOrder];
		final long key = getKey(ngram, startPos, endPos);
		//		final long hash = hash(key, wordOf(key), ngramOrder, currMap);
		//		if (hash < 0) return -1;
		final long index = currMap.getIndexImplicity(key);
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
		if (endPos - startPos == 1) return -1;
		return getOffsetFromRawNgram(ngram, reversed ? startPos : (startPos + 1), reversed ? (endPos - 1) : endPos);
	}

	private long getKey(final int[] ngram, final int startPos, final int endPos) {
		long contextOffset = 0;
		for (int ngramOrder = 0; ngramOrder < endPos - startPos - 1; ++ngramOrder) {
			final int currNgramPos = reversed ? (endPos - ngramOrder - 1) : (startPos + ngramOrder);
			contextOffset = getOffsetForContextEncoding(contextOffset, ngramOrder - 1, ngram[currNgramPos], null);//hash < 0 ? -1L : getOffsetHelp(hash, currEndPos, ngramOrder, null)getIndexHelp(ngram, currStartPos, ngramOrder, currEndPos, hash);
			if (contextOffset == -1L) { return -1; }

		}
		return combineToKey(headWord(ngram, startPos, endPos), contextOffset);
	}

	private int headWord(final int[] ngram, final int startPos, final int endPos) {
		return reversed ? ngram[startPos] : ngram[endPos - 1];
	}

}
