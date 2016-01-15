package edu.berkeley.nlp.lm.map;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
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

	private final boolean storeSuffixOffsets;

	public static <T> HashNgramMap<T> createImplicitWordHashNgramMap(final ValueContainer<T> values, final ConfigOptions opts,
		final LongArray[] numNgramsForEachWord, final boolean reversed) {
		return new HashNgramMap<T>(values, opts, numNgramsForEachWord, reversed);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final LongArray[] numNgramsForEachWord, final boolean reversed) {
		super(values, opts);
		this.reversed = reversed;
		this.maxLoadFactor = opts.hashTableLoadFactor;
		this.storeSuffixOffsets = values.storeSuffixoffsets();
		final int maxNgramOrder = numNgramsForEachWord.length;
		explicitMaps = null;
		isExplicit = false;
		implicitMaps = new ImplicitWordHashMap[maxNgramOrder - 1];
		final long numWords = numNgramsForEachWord[0].size();
		implicitUnigramMap = new UnigramHashMap(numWords, this);
		initCapacities = null;
		final long maxSize = getMaximumSize(numNgramsForEachWord);

		// a little ugly: store word ranges for all orders in the same array to increase cache locality
		// also, if we can, store two ints per long for cache locality
		final boolean fitsInInt = maxSize < Integer.MAX_VALUE;
		final int logicalNumRangeEntries = (maxNgramOrder - 1) * (int) numWords;
		final long[] wordRanges = new long[fitsInInt ? (logicalNumRangeEntries / 2 + logicalNumRangeEntries % 2) : logicalNumRangeEntries];
		values.setMap(this);
		values.setSizeAtLeast(numWords, 0);
		for (int ngramOrder = 1; ngramOrder < maxNgramOrder; ++ngramOrder) {
			final long numNgramsForPreviousOrder = ngramOrder == 1 ? numWords : implicitMaps[ngramOrder - 2].getCapacity();
			implicitMaps[ngramOrder - 1] = new ImplicitWordHashMap(numNgramsForEachWord[ngramOrder], wordRanges, ngramOrder, maxNgramOrder - 1,
				numNgramsForPreviousOrder, (int) numWords, this, fitsInInt, !opts.storeRankedProbBackoffs);
			values.setSizeAtLeast(implicitMaps[ngramOrder - 1].getCapacity(), ngramOrder);
		}
	}

	private long getMaximumSize(final LongArray[] numNgramsForEachWord) {
		long max = Long.MIN_VALUE;
		for (int ngramOrder = 0; ngramOrder < numNgramsForEachWord.length; ++ngramOrder) {
			max = Math.max(max, getSizeOfOrder(numNgramsForEachWord[ngramOrder]));
		}
		return max;
	}

	private long getSizeOfOrder(final LongArray numNgramsForEachWord) {
		long currStart = 0;
		for (int w = (0); w < numNgramsForEachWord.size(); ++w) {

			currStart += getRangeSizeForWord(numNgramsForEachWord, w);

		}
		return currStart;
	}

	/**
	 * @param numNgramsForEachWord
	 * @param w
	 * @return
	 */
	long getRangeSizeForWord(final LongArray numNgramsForEachWord, int w) {
		final long numNgrams = numNgramsForEachWord.get(w);
		final long rangeSize = numNgrams <= 3 ? numNgrams : Math.round(numNgrams * 1.0 / maxLoadFactor);
		return rangeSize;
	}

	/**
	 * Note: Explicit HashNgramMap can grow beyond maxNgramOrder
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
		this.storeSuffixOffsets = values.storeSuffixoffsets();
		this.maxLoadFactor = opts.hashTableLoadFactor;
		implicitMaps = null;
		implicitUnigramMap = null;
		isExplicit = true;
		explicitMaps = new ExplicitWordHashMap[maxNgramOrder];
		initCapacities = new long[maxNgramOrder];
		Arrays.fill(initCapacities, 100);
		values.setMap(this);
	}

	private HashNgramMap(final ValueContainer<T> values, final ConfigOptions opts, final long[] newCapacities, final boolean reversed,
		final ExplicitWordHashMap[] partialMaps) {
		super(values, opts);
		this.reversed = reversed;
		this.storeSuffixOffsets = values.storeSuffixoffsets();
		this.maxLoadFactor = opts.hashTableLoadFactor;
		implicitMaps = null;
		implicitUnigramMap = null;
		isExplicit = true;
		explicitMaps = Arrays.copyOf(partialMaps, newCapacities.length);
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
		return putHelp(ngram, startPos, endPos, val, false);

	}

	/**
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param val
	 * @return
	 */
	private long putHelp(final int[] ngram, final int startPos, final int endPos, final T val, final boolean forcedNew) {
		final int ngramOrder = endPos - startPos - 1;
		HashMap map = getHashMapForOrder(ngramOrder);
		if (!forcedNew && map instanceof ExplicitWordHashMap && map.getLoadFactor() >= maxLoadFactor) {
			rehash(ngramOrder, map.getCapacity() * 3 / 2, 1);
			map = getHashMapForOrder(ngramOrder);
		}
		final long key = getKey(ngram, startPos, endPos);
		if (key < 0) return -1L;
		return putHelp(map, ngram, startPos, endPos, key, val, forcedNew);
	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private HashMap getHashMapForOrder(final int ngramOrder) {
		HashMap map = getMap(ngramOrder);
		if (map == null) {
			final long newCapacity = initCapacities[ngramOrder];
			assert newCapacity >= 0 : "Bad capacity " + newCapacity + " for order " + ngramOrder;
			map = initMap(newCapacity, ngramOrder);
		}

		return map;
	}

	/**
	 * Warning: does not rehash if load factor is exceeded, must call
	 * rehashIfNecessary explicitly. This is so that the offsets returned remain
	 * valid. Basically, you should not use this function unless you really know
	 * what you're doing.
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
		return putHelp(map, ngram, startPos, endPos, key, val, false);
	}

	/**
	 * Warning: does not rehash if load factor is exceeded, must call
	 * rehashIfNecessary explicitly. This is so that the offsets returned remain
	 * valid. Basically, you should not use this function unless you really know
	 * what you're doing.
	 * 
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param contextOffset
	 * @param val
	 * @return
	 */
	public long putWithOffsetAndSuffix(final int[] ngram, final int startPos, final int endPos, final long contextOffset, final long suffixOffset, final T val) {
		final int ngramOrder = endPos - startPos - 1;
		final long key = combineToKey(ngram[endPos - 1], contextOffset);
		final HashMap map = getHashMapForOrder(ngramOrder);
		return putHelpWithSuffixIndex(map, ngram, startPos, endPos, key, val, false, suffixOffset);
	}

	public void rehashIfNecessary(int num) {
		if (explicitMaps == null) return;
		for (int ngramOrder = 0; ngramOrder < explicitMaps.length; ++ngramOrder) {
			if (explicitMaps[ngramOrder] == null) 
				initCapacities[ngramOrder] = Math.max(100, num) * 3/2;
			else if (explicitMaps[ngramOrder].getLoadFactor(num) >= maxLoadFactor) {
				rehash(ngramOrder, (explicitMaps[ngramOrder].getCapacity() + num) * 3 / 2, num);
				return;
			}
		}

	}

	private long putHelp(final HashMap map, final int[] ngram, final int startPos, final int endPos, final long key, final T val, final boolean forcedNew) {
		final long suffixIndex = storeSuffixOffsets ? getSuffixOffset(ngram, startPos, endPos) : -1L;
		return putHelpWithSuffixIndex(map, ngram, startPos, endPos, key, val, forcedNew, suffixIndex);

	}

	/**
	 * @param map
	 * @param ngram
	 * @param startPos
	 * @param endPos
	 * @param key
	 * @param val
	 * @param forcedNew
	 * @param suffixIndex
	 * @return
	 */
	private long putHelpWithSuffixIndex(final HashMap map, final int[] ngram, final int startPos, final int endPos, final long key, final T val,
		final boolean forcedNew, final long suffixIndex) {
		final int ngramOrder = endPos - startPos - 1;
		final long oldSize = map.size();
		final long index = map.put(key);

		final boolean addWorked = values.add(ngram, startPos, endPos, ngramOrder, index, contextOffsetOf(key), wordOf(key), val, suffixIndex,
			map.size() > oldSize || forcedNew);
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
		final int[] ret = new int[Math.max(1, contextOrder + 2)];
		getNgramFromContextEncodingHelp(contextOffset, contextOrder, word, ret);
		return ret;
	}

	/**
	 * @param contextOffset
	 * @param contextOrder
	 * @param word
	 * @param scratch
	 * @return
	 */
	private void getNgramFromContextEncodingHelp(final long contextOffset, final int contextOrder, final int word, final int[] scratch) {
		if (contextOrder < 0) {
			scratch[0] = word;
		} else {
			long contextOffset_ = contextOffset;
			int word_ = word;
			scratch[reversed ? 0 : (scratch.length - 1)] = word_;
			for (int i = 0; i <= contextOrder; ++i) {
				final int ngramOrder = contextOrder - i;
				final long key = getKey(contextOffset_, ngramOrder);
				contextOffset_ = contextOffsetOf(key);

				word_ = wordOf(key);
				scratch[reversed ? (i + 1) : (scratch.length - i - 2)] = word_;
			}
		}

	}

	public int getNextWord(final long offset, final int ngramOrder) {
		return wordOf(getKey(offset, ngramOrder));
	}

	public long getNextContextOffset(final long offset, final int ngramOrder) {
		return contextOffsetOf(getKey(offset, ngramOrder));
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

	public int getFirstWordForOffset(final long offset, final int ngramOrder) {
		final long key = getMap(ngramOrder).getKey(offset);
		if (ngramOrder == 0)
			return wordOf(key);
		else
			return getFirstWordForOffset(contextOffsetOf(key), ngramOrder - 1);
	}

	public int getLastWordForOffset(final long offset, final int ngramOrder) {
		final long key = getMap(ngramOrder).getKey(offset);
		return wordOf(key);
	}

	public int[] getNgramForOffset(final long offset, final int ngramOrder) {
		final int[] ret = new int[ngramOrder + 1];
		return getNgramForOffset(offset, ngramOrder, ret);
	}

	public int[] getNgramForOffset(final long offset, final int ngramOrder, final int[] ret) {
		long offset_ = offset;
		for (int i = 0; i <= ngramOrder; ++i) {
			final long key = getMap(ngramOrder - i).getKey(offset_);
			offset_ = contextOffsetOf(key);
			final int word_ = wordOf(key);
			ret[reversed ? (i) : (ngramOrder - i)] = word_;
		}
		return ret;
	}

	/**
	 * @param contextOffset_
	 * @param contextOrder
	 * @param word
	 * @param logFailure
	 * @return
	 */
	private long getOffsetForContextEncoding(final long contextOffset_, final int contextOrder, final int word, @OutputParameter final T outputVal) {
		if (word < 0) return -1;
		final int ngramOrder = contextOrder + 1;
		final long contextOffset = contextOffset_ >= 0 ? contextOffset_ : 0;

		final long key = combineToKey(word, contextOffset);
		final long offset = getOffsetHelpFromMap(ngramOrder, key);
		if (outputVal != null && offset >= 0) {
			values.getFromOffset(offset, ngramOrder, outputVal);
		}
		return offset;
	}

	private long getOffsetHelpFromMap(int ngramOrder, long key) {
		if (isExplicit) { return (ngramOrder >= explicitMaps.length || explicitMaps[ngramOrder] == null) ? -1 : explicitMaps[ngramOrder].getOffset(key); }
		return ngramOrder == 0 ? implicitUnigramMap.getOffset(key) : implicitMaps[ngramOrder - 1].getOffset(key);
	}

	private void rehash(final int changedNgramOrder, final long newCapacity, final int numAdding) {
		assert isExplicit;
		final long[] newCapacities = new long[explicitMaps.length];
		Arrays.fill(newCapacities, -1L);

		assert changedNgramOrder >= 0;
		long largestCapacity = 0L;
		for (int ngramOrder = 0; ngramOrder < explicitMaps.length; ++ngramOrder) {
			if (explicitMaps[ngramOrder] == null) break;
			if (ngramOrder < changedNgramOrder) {
				newCapacities[ngramOrder] = explicitMaps[ngramOrder].getCapacity();
			} else if (ngramOrder == changedNgramOrder) {
				newCapacities[ngramOrder] = newCapacity;

			} else {
				newCapacities[ngramOrder] = explicitMaps[ngramOrder].getLoadFactor(numAdding) >= maxLoadFactor / 2 ? ((explicitMaps[ngramOrder].getCapacity() + numAdding) * 3 / 2)
					: explicitMaps[ngramOrder].getCapacity();
				largestCapacity = Math.max(largestCapacity, newCapacities[ngramOrder]);
			}
			assert newCapacities[ngramOrder] >= 0 : "Bad capacity " + newCapacities[ngramOrder];
		}
		final ValueContainer<T> newValues = values.createFreshValues(newCapacities);
		final HashNgramMap<T> newMap = new HashNgramMap<T>(newValues, opts, newCapacities, reversed, Arrays.copyOf(explicitMaps, changedNgramOrder));

		for (int ngramOrder = 0; ngramOrder < explicitMaps.length; ++ngramOrder) {
			final ExplicitWordHashMap currHashMap = explicitMaps[ngramOrder];
			if (currHashMap == null) {
				// We haven't initialized this map yet, but make sure there is enough space when we do.
				initCapacities[ngramOrder] = largestCapacity;
				continue;
			}
			final ExplicitWordHashMap newHashMap = (ExplicitWordHashMap) newMap.getHashMapForOrder(ngramOrder);
			final T val = values.getScratchValue();
			final int[] scratchArray = new int[ngramOrder + 1];
			for (long actualIndex = 0; actualIndex < currHashMap.getCapacity(); ++actualIndex) {

				final long key = currHashMap.getKey(actualIndex);
				if (currHashMap.isEmptyKey(key)) continue;
				getNgramFromContextEncodingHelp(contextOffsetOf(key), ngramOrder - 1, wordOf(key), scratchArray);
				final long newKey = newMap.getKey(scratchArray, 0, scratchArray.length);
				assert newKey >= 0 : "Failure for old n-gram " + Arrays.toString(scratchArray) + " :: " + newKey;
				final long index = newHashMap.put(newKey);
				assert index >= 0;

				final long suffixIndex = storeSuffixOffsets ? newMap.getSuffixOffset(scratchArray, 0, scratchArray.length) : -1L;
				assert !storeSuffixOffsets || suffixIndex >= 0 : "Could not find suffix offset for " + Arrays.toString(scratchArray);

				values.getFromOffset(actualIndex, ngramOrder, val);
				final boolean addWorked = newMap.values.add(scratchArray, 0, scratchArray.length, ngramOrder, index, contextOffsetOf(newKey), wordOf(newKey),
					val, suffixIndex, true);
				assert addWorked;

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
		if (currMap == null) return -1;
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
		final long offset = getOffsetFromRawNgram(ngram, reversed ? startPos : (startPos + 1), reversed ? (endPos - 1) : endPos);
		return offset;
	}

	/**
	 * Gets the offset of the context for an n-gram (represented by offset)
	 * 
	 * @param offset
	 * @return
	 */
	public long getPrefixOffset(final long offset, final int ngramOrder) {
		if (ngramOrder == 0) return -1;
		return contextOffsetOf(getKey(offset, ngramOrder));
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
		final HashMap map = getMap(ngramOrder);
		if (map == null)
			return Collections.emptyList();
		else
			return Iterators.able(new Iterators.Transform<Long, Entry<T>>(map.keys().iterator())
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

	public Iterable<Long> getNgramOffsetsForOrder(final int ngramOrder) {
		final HashMap map = getMap(ngramOrder);
		if (map == null)
			return Collections.emptyList();
		else
			return map.keys();
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

	public long getTotalSize() {
		long ret = 0L;
		for (int ngramOrder = 0; ngramOrder < getMaxNgramOrder(); ++ngramOrder) {
			final HashMap currMap = getMap(ngramOrder);
			if (currMap == null) break;
			ret += currMap.size();

		}
		return ret;
	}

	@Override
	public CustomWidthArray getValueStoringArray(final int ngramOrder) {
		return (ngramOrder == 0 || isExplicit) ? null : implicitMaps[ngramOrder - 1].keys;
	}

	@Override
	public void clearStorage() {
		if (implicitMaps != null) {
			for (int i = 0; i < implicitMaps.length; ++i) {
				implicitMaps[i] = null;
			}
		}

		if (explicitMaps != null) {
			for (int i = 0; i < explicitMaps.length; ++i) {
				explicitMaps[i] = null;
			}
		}

	}

	double getLoadFactor() {
		return maxLoadFactor;
	}
}
