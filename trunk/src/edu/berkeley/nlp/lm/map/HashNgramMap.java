package edu.berkeley.nlp.lm.map;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.values.ValueContainer;

/**
 * 
 * @author adampauls
 * 
 * @param <T>
 */
public final class HashNgramMap<T> extends AbstractNgramMap<T> implements ContextEncodedNgramMap<T>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@PrintMemoryCount
	private ExplicitWordHashMap[] explicitMaps;

	@PrintMemoryCount
	private final ImplicitWordHashMap[] implicitMaps;

	@PrintMemoryCount
	private final UnigramHashMap implicitUnigramMap;

	private long[] initCapacities;

	private final double maxLoadFactor;

	private final boolean isExplicit;

	private final boolean reversed;

	public static <T> HashNgramMap<T> createImplicitWordHashNgramMap(final ValueContainer<T> values, final ConfigOptions opts,
		final LongArray[] numNgramsForEachWord, final boolean reversed) {
		return new HashNgramMap<T>(values, opts, numNgramsForEachWord, reversed);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final LongArray[] numNgramsForEachWord, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		final int maxNgramOrder = numNgramsForEachWord.length;
		explicitMaps = null;
		isExplicit = false;
		implicitMaps = new ImplicitWordHashMap[maxNgramOrder - 1];
		final long numWords = numNgramsForEachWord[0].size();
		implicitUnigramMap = new UnigramHashMap(numWords);
		initCapacities = null;
		final long[] wordRanges = new long[(maxNgramOrder - 1) * (int) numWords];
		for (int ngramOrder = 1; ngramOrder < maxNgramOrder; ++ngramOrder) {
			final long numNgramsForPreviousOrder = ngramOrder == 1 ? numWords : implicitMaps[ngramOrder - 2].getCapacity();
			implicitMaps[ngramOrder - 1] = new ImplicitWordHashMap(numNgramsForEachWord[ngramOrder], maxLoadFactor, wordRanges, ngramOrder, maxNgramOrder - 1,
				numNgramsForPreviousOrder, (int) numWords);
			values.setSizeAtLeast(implicitMaps[ngramOrder - 1].getCapacity(), ngramOrder);
		}
		values.setMap(this);
	}

	/**
	 * Note: Explicint HashNgramMap can grow beyond maxNgramOrder
	 * 
	 * @param <T>
	 * @param values
	 * @param opts
	 * @param maxNgramOrder
	 * @param reversed
	 * @return
	 */
	public static <T> HashNgramMap<T> createExplicitWordHashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final int maxNgramOrder,
		final boolean reversed) {
		return new HashNgramMap<T>(values, opts, maxNgramOrder, reversed);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final int maxNgramOrder, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		implicitMaps = null;
		implicitUnigramMap = null;
		isExplicit = true;
		explicitMaps = new ExplicitWordHashMap[maxNgramOrder];
		initCapacities = new long[maxNgramOrder];
		Arrays.fill(initCapacities, 100);
		values.setMap(this);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final long[] newCapacities, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		implicitMaps = null;
		implicitUnigramMap = null;
		isExplicit = true;
		explicitMaps = new ExplicitWordHashMap[newCapacities.length];
		this.initCapacities = newCapacities;
		values.setMap(this);
	}

	/**
	 * @param values
	 * @param newCapacities
	 * @param ngramOrder
	 * @return
	 */
	private ExplicitWordHashMap initMap(final long newCapacity, final int ngramOrder) {
		final ExplicitWordHashMap newMap = new ExplicitWordHashMap(newCapacity);
		explicitMaps[ngramOrder] = newMap;
		values.setSizeAtLeast(explicitMaps[ngramOrder].getCapacity(), ngramOrder);
		return newMap;
	}

	@Override
	public long put(final int[] ngram, final int startPos, final int endPos, final T val) {
		final int ngramOrder = endPos - startPos - 1;
		HashMap map = getHashMapForOrder(ngramOrder);
		if (map instanceof ExplicitWordHashMap && map.getLoadFactor() >= maxLoadFactor) {
			rehash(ngramOrder, map.getCapacity() * 3 / 2);
			map = getHashMapForOrder(ngramOrder);
		}
		final long key = getKey(ngram, startPos, endPos);
		if (key < 0) return -1L;
		return putHelp(map, ngram, startPos, endPos, key, val);

	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private HashMap getHashMapForOrder(final int ngramOrder) {
		HashMap map = getMap(ngramOrder);
		if (map == null) {
			map = initMap(initCapacities[ngramOrder], ngramOrder);
		}

		return map;
	}

	/**
	 * Warning: does not rehash if load factor is exceeded, must call
	 * rehashIfNecessary explicitly
	 * 
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param contextOffset
	 * @param val
	 * @return
	 */
	public long putWithOffset(final int[] ngram, final int startPos, final int endPos, final long contextOffset, final T val) {
		final int ngramOrder = endPos - startPos - 1;
		final long key = combineToKey(ngram[endPos - 1], contextOffset);
		final HashMap map = getHashMapForOrder(ngramOrder);
		return putHelp(map, ngram, startPos, endPos, key, val);
	}

	public void rehashIfNecessary() {
		if (explicitMaps == null) return;
		boolean rehash = false;
		for (int ngramOrder = 0; ngramOrder < explicitMaps.length; ++ngramOrder) {
			if (explicitMaps[ngramOrder] == null) continue;
			rehash |= (explicitMaps[ngramOrder].getLoadFactor() >= maxLoadFactor);
		}
		if (rehash) {
			rehash(-1, -1);
		}
	}

	private long putHelp(final HashMap map, final int[] ngram, final int startPos, final int endPos, final long key, final T val) {
		final int ngramOrder = endPos - startPos - 1;
		final long oldSize = map.size();
		final long index = map.put(key);

		final long suffixIndex = getSuffixOffset(ngram, startPos, endPos);
		final boolean addWorked = values.add(ngram, startPos, endPos, ngramOrder, index, contextOffsetOf(key), wordOf(key), val, suffixIndex,
			map.size() > oldSize);
		if (!addWorked) return -1;

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
	public int[] getNgramFromContextEncoding(final long contextOffset, final int contextOrder, final int word) {
		if (contextOrder < 0) return new int[] { word };
		final int[] ret = new int[contextOrder + 2];
		long contextOffset_ = contextOffset;
		int word_ = word;
		ret[reversed ? 0 : (ret.length - 1)] = word_;
		for (int i = 0; i <= contextOrder; ++i) {
			final int ngramOrder = contextOrder - i;
			final long key = getKey(contextOffset_, ngramOrder);
			contextOffset_ = AbstractNgramMap.contextOffsetOf(key);

			word_ = AbstractNgramMap.wordOf(key);
			ret[reversed ? (i + 1) : (ret.length - i - 2)] = word_;
		}
		return ret;
	}

	public int getNextWord(final long offset, final int ngramOrder) {
		return AbstractNgramMap.wordOf(getKey(offset, ngramOrder));
	}

	public long getNextContextOffset(final long offset, final int ngramOrder) {
		return AbstractNgramMap.contextOffsetOf(getKey(offset, ngramOrder));
	}

	/**
	 * Gets the "key" (word + context offset) for a given offset
	 * 
	 * @param contextOffset_
	 * @param ngramOrder
	 * @return
	 */
	private long getKey(final long offset, final int ngramOrder) {
		return getMap(ngramOrder).getKey(offset);
	}

	public int[] getNgramForOffset(final long offset, final int ngramOrder) {
		final int[] ret = new int[ngramOrder + 1];
		long offset_ = offset;
		for (int i = 0; i <= ngramOrder; ++i) {
			final long key = getMap(ngramOrder - i).getKey(offset_);
			offset_ = AbstractNgramMap.contextOffsetOf(key);
			final int word_ = AbstractNgramMap.wordOf(key);
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
		final long offset = getOffsetHelpFromMap(ngramOrder, key);
		return offset;
	}

	private long getOffsetHelpFromMap(int ngramOrder, long key) {
		if (isExplicit) return explicitMaps[ngramOrder].getOffset(key);
		return ngramOrder == 0 ? implicitUnigramMap.getOffset(key) : implicitMaps[ngramOrder - 1].getOffset(key);
	}

	private void rehash(final int changedNgramOrder, final long newCapacity) {
		assert isExplicit;
		final ValueContainer<T> newValues = values.createFreshValues();
		final long[] newCapacities = new long[explicitMaps.length];
		Arrays.fill(newCapacities, -1L);

		boolean growing = false;
		for (int ngramOrder = 0; ngramOrder < explicitMaps.length; ++ngramOrder) {
			if (explicitMaps[ngramOrder] == null) break;
			if (changedNgramOrder < 0) {
				if ((growing && explicitMaps[ngramOrder].getLoadFactor() >= maxLoadFactor / 2) || explicitMaps[ngramOrder].getLoadFactor() >= maxLoadFactor) {
					growing = true;
					newCapacities[ngramOrder] = explicitMaps[ngramOrder].getCapacity() * 3 / 2;
				} else {
					newCapacities[ngramOrder] = explicitMaps[ngramOrder].getCapacity();
				}
			} else {
				newCapacities[ngramOrder] = ngramOrder == changedNgramOrder ? newCapacity : explicitMaps[ngramOrder].getCapacity();
			}
		}
		final HashNgramMap<T> newMap = new HashNgramMap<T>(newValues, opts, newCapacities, reversed);

		for (int ngramOrder = 0; ngramOrder < explicitMaps.length; ++ngramOrder) {
			final HashMap currMap = explicitMaps[ngramOrder];
			if (currMap == null) continue;
			for (long actualIndex = 0; actualIndex < currMap.getCapacity(); ++actualIndex) {

				final long key = currMap.getKey(actualIndex);
				if (currMap.isEmptyKey(key)) continue;
				final int[] ngram = getNgramFromContextEncoding(AbstractNgramMap.contextOffsetOf(key), ngramOrder - 1, AbstractNgramMap.wordOf(key));

				final T val = values.getScratchValue();
				values.getFromOffset(actualIndex, ngramOrder, val);

				newMap.put(ngram, 0, ngram.length, val);

			}
			values.clearStorageForOrder(ngramOrder);
		}
		System.arraycopy(newMap.explicitMaps, 0, explicitMaps, 0, newMap.explicitMaps.length);
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
		if (ngramOrder >= getMaxNgramOrder()) return -1;
		final long key = getKey(ngram, startPos, endPos);
		if (key < 0) return -1;
		final HashMap currMap = getMap(ngramOrder);
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
		for (int ngramOrder = 0; ngramOrder < getMaxNgramOrder(); ++ngramOrder) {
			final HashMap currMap = getMap(ngramOrder);
			if (currMap == null) break;
			values.trimAfterNgram(ngramOrder, currMap.getCapacity());
			Logger.logss("Load factor for " + (ngramOrder + 1) + ": " + currMap.getLoadFactor());
		}
		values.trim();
	}

	/**
	 * @param ngram
	 * @param endPos
	 * @return
	 */
	private long getSuffixOffset(final int[] ngram, final int startPos, final int endPos) {
		if (endPos - startPos == 1) return 0;
		return getOffsetFromRawNgram(ngram, reversed ? startPos : (startPos + 1), reversed ? (endPos - 1) : endPos);
	}

	/**
	 * Gets the offset of the context for an n-gram (represented by offset)
	 * 
	 * @param offset
	 * @return
	 */
	public long getPrefixOffset(final long offset, final int ngramOrder) {
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
		return explicitMaps == null ? (implicitMaps.length + 1) : explicitMaps.length;
	}

	@Override
	public long getNumNgrams(final int ngramOrder) {
		return getMap(ngramOrder).size();
	}

	@Override
	public Iterable<Entry<T>> getNgramsForOrder(final int ngramOrder) {
		return Iterators.able(new Iterators.Transform<Long, Entry<T>>(getMap(ngramOrder).keys().iterator())
		{

			@Override
			protected Entry<T> transform(final Long next) {
				final long offset = next;
				final T val = values.getScratchValue();
				values.getFromOffset(offset, ngramOrder, val);
				return new Entry<T>(getNgramForOffset(offset, ngramOrder), val);
			}
		});
	}

	private HashMap getMap(int ngramOrder) {
		if (explicitMaps == null) { return ngramOrder == 0 ? implicitUnigramMap : implicitMaps[ngramOrder - 1]; }
		if (ngramOrder >= explicitMaps.length) {
			int oldLength = explicitMaps.length;
			explicitMaps = Arrays.copyOf(explicitMaps, explicitMaps.length * 2);
			initCapacities = Arrays.copyOf(initCapacities, initCapacities.length * 2);
			Arrays.fill(initCapacities, oldLength, initCapacities.length, 100);
		}
		return explicitMaps[ngramOrder];
	}

	public boolean isReversed() {
		return reversed;
	}

	@Override
	public boolean wordHasBigrams(final int word) {
		return getMaxNgramOrder() < 2 ? false : (explicitMaps == null ? implicitMaps[0].hasContexts(word) : explicitMaps[1].hasContexts(word));
	}

	@Override
	public boolean contains(final int[] ngram, final int startPos, final int endPos) {
		return getOffsetFromRawNgram(ngram, startPos, endPos) >= 0;
	}

	@Override
	public T get(int[] ngram, int startPos, int endPos) {
		final long offset = getOffsetFromRawNgram(ngram, startPos, endPos);
		if (offset < 0) {
			return null;
		} else {
			final T val = values.getScratchValue();
			values.getFromOffset(offset, endPos - startPos - 1, val);
			return val;
		}
	}

}
