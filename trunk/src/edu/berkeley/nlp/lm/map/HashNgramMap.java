package edu.berkeley.nlp.lm.map;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.MurmurHash;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.ValueContainer;

public final class HashNgramMap<T> extends AbstractNgramMap<T> implements ContextEncodedNgramMap<T>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@PrintMemoryCount
	private final HashMap[] maps;

	private final long[] initCapacities;

	private final double maxLoadFactor;

	private final boolean reversed;

	public static <T> HashNgramMap<T> createImplicitWordHashNgramMap(ValueContainer<T> values, ConfigOptions opts, LongArray[] numNgramsForEachWord,
		boolean reversed) {
		return new HashNgramMap<T>(values, opts, numNgramsForEachWord, reversed);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final LongArray[] numNgramsForEachWord, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		maps = new HashMap[numNgramsForEachWord.length];
		initCapacities = null;
		for (int ngramOrder = 0; ngramOrder < numNgramsForEachWord.length; ++ngramOrder) {
			maps[ngramOrder] = (ngramOrder == 0) ? new UnigramHashMap(numNgramsForEachWord[ngramOrder].size()) : new ImplicitWordHashMap(
				numNgramsForEachWord[ngramOrder], maxLoadFactor);
			values.setSizeAtLeast(maps[ngramOrder].getCapacity(), ngramOrder);
		}
		values.setMap(this);
	}

	public static <T> HashNgramMap<T> createExplicitWordHashNgramMap(ValueContainer<T> values, ConfigOptions opts, final int maxNgramOrder, boolean reversed) {
		return new HashNgramMap<T>(values, opts, maxNgramOrder, reversed);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final int maxNgramOrder, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		maps = new HashMap[maxNgramOrder];
		initCapacities = new long[maxNgramOrder];
		Arrays.fill(initCapacities, 100);
		values.setMap(this);
	}

	private HashNgramMap(ValueContainer<T> values, ConfigOptions opts, long[] newCapacities, boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		maps = new ExplicitWordHashMap[newCapacities.length];
		this.initCapacities = newCapacities;
		values.setMap(this);
	}

	/**
	 * @param values
	 * @param newCapacities
	 * @param ngramOrder
	 */
	private void initMap(long newCapacity, int ngramOrder) {
		maps[ngramOrder] = new ExplicitWordHashMap(newCapacity);
		values.setSizeAtLeast(maps[ngramOrder].getCapacity(), ngramOrder);
	}

	@Override
	public long put(final int[] ngram, int startPos, int endPos, final T val) {
		final int ngramOrder = endPos - startPos - 1;
		HashMap map = maps[ngramOrder];
		if (map == null) {
			initMap(initCapacities[ngramOrder], ngramOrder);
			map = maps[ngramOrder];
		}
		if (map instanceof ExplicitWordHashMap && map.getLoadFactor() >= maxLoadFactor) {
			rehash(ngramOrder, map.getCapacity() * 3 / 2);
			map = maps[ngramOrder];
		}
		final long key = getKey(ngram, startPos, endPos);
		if (key < 0) return -1L;
		long oldSize = map.size();
		final long index = map.put(key);

		final long suffixIndex = getSuffixOffset(ngram, startPos, endPos);
		values.add(ngram, startPos, endPos, ngramOrder, index, contextOffsetOf(key), wordOf(key), val, suffixIndex, map.size() > oldSize);
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

	@Override
	public int[] getNgramFromContextEncoding(long contextOffset, int contextOrder, int word) {
		if (contextOrder < 0) return new int[] { word };
		int[] ret = new int[contextOrder + 2];
		long contextOffset_ = contextOffset;
		int word_ = word;
		ret[reversed ? 0 : (ret.length - 1)] = word_;
		for (int i = 0; i <= contextOrder; ++i) {
			final int ngramOrder = contextOrder - i;
			long key = getKey(contextOffset_, ngramOrder);
			contextOffset_ = AbstractNgramMap.contextOffsetOf(key);

			word_ = AbstractNgramMap.wordOf(key);
			ret[reversed ? (i + 1) : (ret.length - i - 2)] = word_;
		}
		return ret;
	}

	public int getNextWord(long offset, final int ngramOrder) {
		return AbstractNgramMap.wordOf(getKey(offset, ngramOrder));
	}

	public long getNextContextOffset(long offset, final int ngramOrder) {
		return AbstractNgramMap.contextOffsetOf(getKey(offset, ngramOrder));
	}

	/**
	 * Gets the "key" (word + context offset) for a given offset
	 * 
	 * @param contextOffset_
	 * @param ngramOrder
	 * @return
	 */
	private long getKey(long offset, final int ngramOrder) {
		return maps[ngramOrder].getKey(offset);
	}

	public int[] getNgramForOffset(long offset, int ngramOrder) {
		int[] ret = new int[ngramOrder + 1];
		long offset_ = offset;
		for (int i = 0; i <= ngramOrder; ++i) {
			long key = maps[ngramOrder - i].getKey(offset_);
			offset_ = AbstractNgramMap.contextOffsetOf(key);
			int word_ = AbstractNgramMap.wordOf(key);
			ret[reversed ? (i) : (ret.length - i - 1)] = word_;
		}
		return ret;
	}

	/**
	 * @param contextOffset_
	 * @param contextOrder
	 * @param word
	 * @return
	 */
	private long getOffsetForContextEncoding(final long contextOffset_, final int contextOrder, final int word, @OutputParameter final T outputVal) {
		final int ngramOrder = contextOrder + 1;
		final long offset = getOffsetHelp(contextOffset_, word, ngramOrder);
		if (offset >= 0 && outputVal != null) {
			values.getFromOffset(offset, ngramOrder, outputVal);
		}
		return offset;
	}

	/**
	 * @param contextOffset_
	 * @param word
	 * @param ngramOrder
	 * @return
	 */
	private long getOffsetHelp(final long contextOffset_, final int word, final int ngramOrder) {
		final long contextOffset = Math.max(contextOffset_, 0);

		final long key = combineToKey(word, contextOffset);
		final HashMap map = maps[ngramOrder];
		final long offset = map.getOffset(key);
		return offset;
	}

	private void rehash(final int changedNgramOrder, final long newCapacity) {
		final ValueContainer<T> newValues = values.createFreshValues();
		final long[] newCapacities = new long[maps.length];
		Arrays.fill(newCapacities, -1L);

		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
			if (maps[ngramOrder] == null) break;
			newCapacities[ngramOrder] = ngramOrder == changedNgramOrder ? newCapacity : maps[ngramOrder].getCapacity();
		}
		final HashNgramMap<T> newMap = new HashNgramMap<T>(newValues, opts, newCapacities, reversed);

		for (int ngramOrder = 0; ngramOrder < maps.length; ++ngramOrder) {
			final HashMap currMap = maps[ngramOrder];
			if (currMap == null) continue;
			for (long actualIndex = 0; actualIndex < currMap.getCapacity(); ++actualIndex) {
				final long key = currMap.getKey(actualIndex);
				if (currMap.isEmptyKey(key)) continue;
				final int[] ngram = getNgramFromContextEncoding(AbstractNgramMap.contextOffsetOf(key), ngramOrder - 1, AbstractNgramMap.wordOf(key));

				final T val = values.getScratchValue();
				values.getFromOffset(actualIndex, ngramOrder, val);

				newMap.put(ngram, 0, ngram.length, val);

			}
		}
		System.arraycopy(newMap.maps, 0, maps, 0, newMap.maps.length);
		values.setFromOtherValues(newValues);
		values.setMap(this);

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
		if (ngramOrder >= maps.length) return -1;
		final HashMap currMap = maps[ngramOrder];
		final long key = getKey(ngram, startPos, endPos);
		if (key < 0) return -1;
		final long index = currMap.getOffset(key);
		return index;
	}

	@Override
	public LmContextInfo getOffsetForNgram(final int[] ngram, final int startPos, final int endPos) {
		final LmContextInfo lmContextInfo = new LmContextInfo();
		for (int start = endPos - 1; start >= startPos; --start) {
			final long offset = getOffsetFromRawNgram(ngram, start, endPos);
			if (offset < 0) break;
			lmContextInfo.offset = offset;
			lmContextInfo.order = endPos - start - 1;
		}
		return lmContextInfo;
	}

	/**
	 * Like {@link #getOffsetForNgram(int[], int, int)}, but assumes that the
	 * full n-gram is in the map (i.e. does not back off to the largest suffix
	 * which is in the model).
	 * 
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @return
	 */
	public long getOffsetForNgramInModel(final int[] ngram, final int startPos, final int endPos) {
		return getOffsetFromRawNgram(ngram, startPos, endPos);
	}

	@Override
	public void handleNgramsFinished(final int justFinishedOrder) {
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

	/**
	 * Gets the offset of the context for an n-gram (represented by offset)
	 * 
	 * @param offset
	 * @return
	 */
	public long getPrefixOffset(long offset, int ngramOrder) {
		if (ngramOrder == 0) return -1;
		return AbstractNgramMap.contextOffsetOf(getKey(offset, ngramOrder));
	}

	private long getKey(final int[] ngram, final int startPos, final int endPos) {
		long contextOffset = 0;
		for (int ngramOrder = 0; ngramOrder < endPos - startPos - 1; ++ngramOrder) {
			final int currNgramPos = reversed ? (endPos - ngramOrder - 1) : (startPos + ngramOrder);
			contextOffset = getOffsetForContextEncoding(contextOffset, ngramOrder - 1, ngram[currNgramPos], null);
			if (contextOffset == -1L) { return -1; }

		}
		return combineToKey(headWord(ngram, startPos, endPos), contextOffset);
	}

	private int headWord(final int[] ngram, final int startPos, final int endPos) {
		return reversed ? ngram[startPos] : ngram[endPos - 1];
	}

	@Override
	public int getMaxNgramOrder() {
		return maps.length;
	}

	@Override
	public long getNumNgrams(int ngramOrder) {
		return maps[ngramOrder].size();
	}

	@Override
	public Iterable<Entry<T>> getNgramsForOrder(final int ngramOrder) {
		return Iterators.able(new Iterators.Transform<Long, Entry<T>>(maps[ngramOrder].keys().iterator())
		{

			@Override
			protected Entry<T> transform(Long next) {
				long offset = next;
				final T val = values.getScratchValue();
				values.getFromOffset(offset, ngramOrder, val);
				return new Entry<T>(getNgramForOffset(offset, ngramOrder), val);
			}
		});
	}

	public boolean isReversed() {
		return reversed;
	}

	@Override
	public boolean wordHasBigrams(int word) {
		return maps[1].hasContexts(word);
	}

}
