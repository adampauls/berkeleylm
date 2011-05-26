package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.values.ValueContainer;

public abstract class BinarySearchNgramMap<T> extends AbstractNgramMap<T> implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected abstract static class InternalSortedMap implements Serializable
	{
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		@PrintMemoryCount
		protected LongArray keys;

		@PrintMemoryCount
		protected int[] wordRangesLow;

		@PrintMemoryCount
		protected int[] wordRangesHigh;

		protected long cachedLastIndex;

		protected int[] cachedLastSuffix;

		public long add(final long key) {
			final InternalSortedMap map = this;
			map.keys.add(key);
			return map.keys.size();
		}

		public long size() {
			return keys.size();
		}

		/**
		 * @param justFinishedOrder
		 * @param currKeys
		 */
		public void buildIndexes(final int numWords) {
			wordRangesLow = new int[numWords];
			final int[] lows = wordRangesLow;
			Arrays.fill(lows, -1);
			wordRangesHigh = new int[numWords];
			final int[] highs = wordRangesHigh;
			long lastWord = -1;
			long keySize = keys.size();
			for (long i = 0; i <= keys.size(); ++i) {
				final long currWord = i == keySize ? -1L : wordOf(keys.get(i));
				if (currWord < 0 || currWord != lastWord) {
					if (lastWord >= 0) highs[(int) lastWord] = (int) i;
					if (currWord >= 0) lows[(int) currWord] = (int) i;
				}
				lastWord = currWord;
			}
		}

		public void init(final long l) {
			keys = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, l);
		}
	}

	InternalSortedMap[] maps;

	protected final boolean synchronize;

	protected final boolean reverseTrie;

	protected long numWords;

	public BinarySearchNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final boolean reverseTrie) {
		super(values, opts);
		maps = new InternalSortedMap[10];
		this.reverseTrie = reverseTrie;
		this.synchronize = opts.numGoogleLoadThreads > 0;

	}

	protected static int compareLongsRaw(final long a, final long b) {
		assert a >= 0;
		assert b >= 0;
		if (a > b) return +1;
		if (a < b) return -1;
		if (a == b) return 0;
		throw new RuntimeException();
	}

	protected void sort(final LongArray array, final long left0, final long right0, final int ngramOrder) {

		long left, right;
		long pivot;
		left = left0;
		right = right0 + 1;

		final long pivotIndex = (left0 + right0) >>> 1;

		pivot = array.get(pivotIndex);//[outerArrayPart(pivotIndex)][innerArrayPart(pivotIndex)];
		swap(pivotIndex, left0, array, ngramOrder);

		do {

			do
				left++;
			while (left <= right0 && compareLongsRaw(array.get(left), pivot) < 0);

			do
				right--;
			while (compareLongsRaw(array.get(right), pivot) > 0);

			if (left < right) {
				swap(left, right, array, ngramOrder);
			}

		} while (left <= right);

		swap(left0, right, array, ngramOrder);

		if (left0 < right) sort(array, left0, right, ngramOrder);
		if (left < right0) sort(array, left, right0, ngramOrder);

	}

	protected void swap(final long a, final long b, final LongArray array, final int ngramOrder) {
		swap(array, a, b);
		values.swap(a, b, ngramOrder);
	}

	protected void swap(final LongArray array, final long a, final long b) {
		final long temp = array.get(a);
		array.set(a, array.get(b));
		array.set(b, temp);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.NgramMap#add(java.util.List, T)
	 */
	@Override
	public long put(final int[] ngram, final T val) {

		final int ngramOrder = ngram.length - 1;
		final int word = reverseTrie ? ngram[0] : ngram[ngram.length - 1];

		final long prefixOffset = reverseTrie ? getPrefixOffset(ngram, 1, ngram.length) : getPrefixOffset(ngram, 0, ngram.length - 1);
		if (prefixOffset < 0) {

		return -1; }

		long newOffset = -1;
		if (synchronize) {
			synchronized (this) {
				newOffset = doAdd(ngram, val, ngramOrder, prefixOffset, word, -1);
			}
		} else {
			newOffset = doAdd(ngram, val, ngramOrder, prefixOffset, word, -1);
		}
		return newOffset;

	}

	/**
	 * @param ngram
	 * @param val
	 * @param ngramOrder
	 * @param newKey
	 */
	private long doAdd(final int[] ngram, final T val, final int ngramOrder, final long contextOffset, final int word, final long suffixOffset) {
		if (ngram.length > maps.length) {
			//			handleNgramsFinished(currNgramOrder);
			maps = Arrays.copyOf(maps, maps.length * 3 / 2);
			//			noWordKeys = Arrays.copyOf(noWordKeys, noWordKeys.length * 3 / 2);
		}
		if (maps[ngramOrder] == null) maps[ngramOrder] = newInternalSortedMap();
		final long newOffset = maps[ngramOrder].add(getKey(word, contextOffset));
		values.add(ngramOrder, maps[ngramOrder].size() - 1, contextOffset, word, val, suffixOffset);
		return newOffset;
	}

	abstract protected InternalSortedMap newInternalSortedMap();

	private long getPrefixOffset(final int[] ngram, final int startPos, final int endPos) {
		if (endPos == startPos) return 0;
		return getPrefixOffsetHelp(ngram, startPos, endPos);
	}

	abstract protected long getPrefixOffsetHelp(int[] ngram, int startPos, int endPos);

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.NgramMap#handleNgramsFinished(int)
	 */
	@Override
	public void handleNgramsFinished(final int justFinishedOrder) {
		final LongArray currKeys = maps[justFinishedOrder - 1].keys;
		final long currSize = currKeys.size();
		sort(currKeys, 0, currSize - 1, justFinishedOrder - 1);
		maps[justFinishedOrder - 1].keys.trim();
		if (justFinishedOrder == 1) numWords = currSize;
		values.trimAfterNgram(justFinishedOrder - 1, currSize);
	}

	protected int compare(final long key_, final int[] phrase, final int startPos, final int endPos) {
		long key = key_;
		if (reverseTrie) {
			for (int pos = startPos; pos < endPos; ++pos) {
				final long firstWordId = phrase[pos];
				final long firstWordInHash = wordOf(key);
				final long suffixIndex = contextOffsetOf(key);
				final int ngramOrder = endPos - pos - 1;
				if (ngramOrder == 0 || firstWordInHash != firstWordId) return compareLongsRaw(firstWordInHash, firstWordId);
				key = maps[ngramOrder - 1].keys.get(suffixIndex);//[outerArrayPart(suffixIndex)][innerArrayPart(suffixIndex)];
			}
		} else {
			for (int pos = endPos - 1; pos >= startPos; --pos) {
				final long firstWordId = phrase[pos];
				final long firstWordInHash = wordOf(key);
				final long suffixIndex = wordOf(key);
				final int ngramOrder = endPos - pos - 1;
				if (ngramOrder == 0 || firstWordInHash != firstWordId) return compareLongsRaw(firstWordInHash, firstWordId);
				key = maps[ngramOrder - 1].keys.get(suffixIndex);//[outerArrayPart(suffixIndex)][innerArrayPart(suffixIndex)];
			}
		}
		return 0;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.NgramMap#trim()
	 */
	@Override
	public void trim() {
		values.trim();

	}

	@Override
	public void initWithLengths(final List<Long> numNGrams) {
		maps = new InternalSortedMap[numNGrams.size()];
		for (int i = 0; i < numNGrams.size(); ++i) {
			maps[i] = newInternalSortedMap();
			final long l = numNGrams.get(i);
			maps[i].init(l);
			values.setSizeAtLeast(l, i);

		}
	}

}
