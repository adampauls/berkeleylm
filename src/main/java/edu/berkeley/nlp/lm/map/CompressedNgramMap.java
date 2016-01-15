package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.bits.VariableLengthBitCompressor;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.values.CompressibleValueContainer;

public class CompressedNgramMap<T> extends AbstractNgramMap<T> implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final int compressedBlockSize;

	private static final int OFFSET_RADIX = 33;

	private static final int WORD_RADIX = 2;

	private final VariableLengthBitCompressor offsetCoder;

	private final VariableLengthBitCompressor wordCoder;

	private final VariableLengthBitCompressor suffixCoder;

	private double totalKeyBitsFinal = 0;

	private double totalValueBitsFinal = 0;

	private double totalBitsFinal = 0;

	private double totalSizeFinal = 0;

	private final int offsetDeltaRadix;

	private final CompressedMap[] maps;

	private final boolean reverseTrie = true;

	private final long[] numNgramsForEachOrder;

	public CompressedNgramMap(final CompressibleValueContainer<T> values, final long[] numNgramsForEachOrder, final ConfigOptions opts) {
		super(values, opts);
		offsetCoder = new VariableLengthBitCompressor(OFFSET_RADIX);
		wordCoder = new VariableLengthBitCompressor(WORD_RADIX);
		this.offsetDeltaRadix = opts.offsetDeltaRadix;
		suffixCoder = new VariableLengthBitCompressor(offsetDeltaRadix);
		this.compressedBlockSize = opts.compressedBlockSize;
		this.numNgramsForEachOrder = numNgramsForEachOrder;
		this.maps = new CompressedMap[numNgramsForEachOrder.length];
		values.setMap(this);

	}

	@Override
	public long getValueAndOffset(final long contextOffset, final int contextNgramOrder, final int word, @OutputParameter final T outputVal) {
		if (word < 0) return -1L;
		final long hash = combineToKey(word, contextOffset);
		final int ngramOrder = contextNgramOrder + 1;
		final LongArray compressedKeys = (maps[ngramOrder]).compressedKeys;
		final long currIndex = decompressSearch(compressedKeys, hash, ngramOrder, outputVal);
		return currIndex;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.NgramMap#add(java.util.List, T)
	 */
	@Override
	public long put(final int[] ngram, final int startPos, final int endPos, final T val) {

		final int ngramOrder = endPos - startPos - 1;
		final int word = reverseTrie ? ngram[startPos] : ngram[endPos - 1];

		final long contextOffset = reverseTrie ? getContextOffset(ngram, startPos + 1, endPos, null) : getContextOffset(ngram, startPos, endPos - 1, null);
		if (contextOffset < 0) return -1;

		CompressedMap map = maps[ngramOrder];
		if (map == null) {
			map = maps[ngramOrder] = new CompressedMap();
			final long l = numNgramsForEachOrder[ngramOrder];
			maps[ngramOrder].init(l);
			values.setSizeAtLeast(l, ngramOrder);
		}
		final long oldSize = map.size();
		final long newOffset = map.add(combineToKey(word, contextOffset));
		final boolean addWorked = values.add(ngram, startPos, endPos, ngramOrder, map.size() - 1, contextOffset, word, val, -1, map.size() == oldSize);
		if (!addWorked) return -1;
		return newOffset;

	}

	private long getContextOffset(final int[] ngram, final int startPos, final int endPos, T val) {
		if (endPos == startPos) return 0;
		long hasValueSuffixIndex = 0;
		if (endPos > startPos) {
			long lastSuffix = 0L;
			for (int ngramOrder = 0; ngramOrder < endPos - startPos; ++ngramOrder) {
				final int firstWord = reverseTrie ? ngram[endPos - ngramOrder - 1] : ngram[startPos + ngramOrder];
				final long key = combineToKey(firstWord, lastSuffix);

				if (maps[ngramOrder] == null) return -1;

				final LongArray compressedKeys = (maps[ngramOrder]).compressedKeys;
				final long currIndex = decompressSearch(compressedKeys, key, ngramOrder, val);
				if (currIndex < 0) return -1;
				lastSuffix = currIndex;
			}
			hasValueSuffixIndex = lastSuffix;
		}
		return hasValueSuffixIndex;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.NgramMap#handleNgramsFinished(int)
	 */
	@Override
	public void handleNgramsFinished(final int justFinishedOrder) {
		final CompressedMap compressedMap = maps[justFinishedOrder - 1];
		if (compressedMap != null) {
			final LongArray currKeys = compressedMap.getUncompressedKeys();
			final long currSize = currKeys.size();
			sort(currKeys, 0, currSize - 1, justFinishedOrder - 1);
			compressedMap.trim();
			values.trimAfterNgram(justFinishedOrder - 1, currSize);
			compress(justFinishedOrder - 1);
		}
	}

	protected static int compareLongsRaw(final long a, final long b) {
		assert a >= 0;
		assert b >= 0;
		if (a > b) return +1;
		if (a < b) return -1;
		if (a == b) return 0;
		throw new RuntimeException();
	}

	private void compress(final int ngramOrder) {
		if (ngramOrder > 0) {
			(maps[ngramOrder]).compressedKeys = compress(maps[ngramOrder].getUncompressedKeys(), maps[ngramOrder].size(), ngramOrder);
			((CompressibleValueContainer<T>) values).clearStorageAfterCompression(ngramOrder);
			maps[ngramOrder].clearUncompressedKeys();
		}
	}

	private LongArray compress(final LongArray uncompressed, final long uncompressedSize, final int ngramOrder) {
		Logger.startTrack("Compressing");
		final LongArray compressedLongArray = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, uncompressedSize >>> 2);

		long uncompressedPos = 0;
		long totalNumKeyBits = 0;
		long totalNumValueBits = 0;
		long currBlock = 0;

		final CompressibleValueContainer<T> compressibleValues = (CompressibleValueContainer<T>) values;
		while (uncompressedPos < uncompressedSize) {
			final BitList currBlockBits = new BitList();
			final long firstKey = uncompressed.get(uncompressedPos);

			if (currBlock++ % 1000 == 0) Logger.logs("On block " + currBlock + " starting at pos " + uncompressedPos);

			currBlockBits.addLong(firstKey);
			final BitList offsetBits = offsetCoder.compress(uncompressedPos);

			final BitList firstValueBits = compressibleValues.getCompressed(uncompressedPos, ngramOrder);
			BitList headerBits = new BitList();
			BitList bodyBits = new BitList();
			long numKeyBits = 0;
			long numValueBits = 0;
			long currUncompressedPos = -1;

			// try compression assuming all words are the same (wordBitOn = false), and if that fails,
			// roll back and try with wordBitOn = true
			OUTER: for (boolean wordBitOn = false, done = false; !done; wordBitOn = true) {
				numKeyBits = 0;
				numValueBits = 0;
				long lastFirstWord = wordOf(firstKey);
				long lastSuffixPart = contextOffsetOf(firstKey);
				headerBits = makeHeader(offsetBits, firstValueBits, wordBitOn);
				bodyBits = new BitList();

				final BitList currBits = new BitList();
				for (currUncompressedPos = uncompressedPos + 1; currUncompressedPos < uncompressedSize; ++currUncompressedPos) {
					final long currKey = uncompressed.get(currUncompressedPos);
					final long currFirstWord = wordOf(currKey);
					final long currSuffixPart = contextOffsetOf(currKey);

					final long wordDelta = currFirstWord - lastFirstWord;
					final long suffixDelta = currSuffixPart - lastSuffixPart;
					currBits.clear();
					if (wordDelta > 0 && !wordBitOn) continue OUTER;
					if (wordBitOn) {
						final BitList keyBits = wordCoder.compress(wordDelta);
						currBits.addAll(keyBits);
						if (wordDelta > 0) {
							final BitList suffixBits = suffixCoder.compress(currSuffixPart);
							currBits.addAll(suffixBits);
						} else {
							final BitList suffixBits = suffixCoder.compress(suffixDelta);
							currBits.addAll(suffixBits);
						}
					} else {

						final BitList suffixBits = suffixCoder.compress(suffixDelta);
						currBits.addAll(suffixBits);
					}

					numKeyBits += currBits.size();
					lastFirstWord = currFirstWord;
					numValueBits += compressValue(ngramOrder, currUncompressedPos, currBits);

					lastSuffixPart = currSuffixPart;
					if (blockFull(currBlockBits, bodyBits, headerBits, currBits)) {
						break;
					}

					bodyBits.addAll(currBits);
				}
				done = true;
			}

			uncompressedPos = currUncompressedPos;

			totalNumKeyBits += numKeyBits;
			totalNumValueBits += numValueBits;

			final int bitLength = bodyBits.size() + headerBits.size();
			assert bitLength <= Short.MAX_VALUE;
			currBlockBits.addShort((short) bitLength);
			currBlockBits.addAll(headerBits);
			currBlockBits.addAll(bodyBits);

			assert currBlockBits.size() < Long.SIZE * compressedBlockSize;
			writeBlockToArray(currBlockBits, compressedLongArray);
		}
		compressedLongArray.trim();

		logCompressionInfo(uncompressedSize, compressedLongArray, totalNumKeyBits, totalNumValueBits);

		Logger.endTrack();
		return compressedLongArray;
	}

	/**
	 * @param blockBits
	 * @param array
	 */
	private void writeBlockToArray(final BitList blockBits, final LongArray array) {
		long curr = 0L;
		for (int i = 0; i <= Long.SIZE * compressedBlockSize; ++i) {
			if (i % Long.SIZE == 0 && i > 0) {
				array.add(curr);
				curr = 0;

			}
			curr = (curr << 1) | ((i >= blockBits.size() || !blockBits.get(i)) ? 0 : 1);
		}
		assert array.size() % compressedBlockSize == 0;
	}

	/**
	 * @param uncompressedSize
	 * @param compressedLongArray
	 * @param keyBits
	 * @param valueBits
	 */
	private void logCompressionInfo(final long uncompressedSize, final LongArray compressedLongArray, final long keyBits, final long valueBits) {
		final double keyAvg = (double) keyBits / uncompressedSize;
		Logger.logss("Key bits " + keyAvg);
		final double valueAvg = (double) valueBits / uncompressedSize;
		Logger.logss("Value bits " + valueAvg);
		final double avg = 64 * (double) compressedLongArray.size() / uncompressedSize;
		Logger.logss("Compressed bits " + avg);
		totalKeyBitsFinal += keyBits;
		totalValueBitsFinal += valueBits;
		totalBitsFinal += compressedLongArray.size();
		totalSizeFinal += uncompressedSize;
		Logger.logss("Total key bits " + totalKeyBitsFinal / totalSizeFinal);
		Logger.logss("Total value bits " + totalValueBitsFinal / totalSizeFinal);
		Logger.logss("Total bits " + 64.0 * totalBitsFinal / totalSizeFinal);
	}

	/**
	 * @param currBits
	 * @param restBits
	 * @param headerBits
	 * @param newBits
	 * @return
	 */
	private boolean blockFull(final BitList currBits, final BitList restBits, final BitList headerBits, final BitList newBits) {
		final int numTotalBitsSize = Short.SIZE;
		final int lengthSoFar = currBits.size() + numTotalBitsSize + headerBits.size() + restBits.size() + newBits.size();
		return lengthSoFar >= Long.SIZE * compressedBlockSize;
	}

	/**
	 * @param ngramOrder
	 * @param valBits
	 * @param currPos
	 * @param newBits
	 * @return
	 */
	private long compressValue(final int ngramOrder, final long currPos, final BitList newBits) {
		final BitList valueBits = ((CompressibleValueContainer<T>) values).getCompressed(currPos, ngramOrder);
		newBits.addAll(valueBits);
		return valueBits.size();
	}

	/**
	 * @param offsetBits
	 * @param firstValueBits
	 * @param wordBitOn
	 * @return
	 */
	private BitList makeHeader(final BitList offsetBits, final BitList firstValueBits, final boolean wordBitOn) {
		BitList headerBits;
		headerBits = new BitList();

		headerBits.addAll(offsetBits);
		headerBits.add(wordBitOn);
		headerBits.addAll(firstValueBits);
		return headerBits;
	}

	/**
	 * searchOffset >= 0 means we are looking for a specific offset and ignore
	 * searchKey if searchOffset >= 0, we return the key, else we return the
	 * offset for searchKey
	 * 
	 * @param compressed
	 * @param pos
	 * @param searchKey
	 * @param ngramOrder
	 * @param outputVal
	 * @param searchOffset
	 * @return
	 */
	private long decompressLinearSearch(final LongArray compressed, final long pos, final long searchKey, final int ngramOrder, final T outputVal,
		final long searchOffset) {

		final long firstKey = compressed.get(pos);
		final BitStream bits = getCompressedBits(compressed, pos + 1);
		final long offset = offsetCoder.decompress(bits);
		final boolean wordBitOn = bits.nextBit();

		int currWord = wordOf(firstKey);
		long currSuffix = contextOffsetOf(firstKey);
		final boolean foundKeyFirst = searchOffset >= 0 ? searchOffset == offset : firstKey == searchKey;

		final CompressibleValueContainer<T> compressibleValues = (CompressibleValueContainer<T>) values;
		compressibleValues.decompress(bits, ngramOrder, !foundKeyFirst, outputVal);
		if (foundKeyFirst) return searchOffset >= 0 ? firstKey : offset;

		long currKey = -1;

		for (int k = 1; !bits.finished(); ++k) {
			int newWord = -1;
			long nextSuffix = -1;

			if (wordBitOn) {
				final int wordDelta = (int) wordCoder.decompress(bits);
				final boolean wordDeltaIsZero = wordDelta == 0;
				final long suffixDelta = suffixCoder.decompress(bits);
				newWord = currWord + wordDelta;
				nextSuffix = wordDeltaIsZero ? (currSuffix + suffixDelta) : suffixDelta;
			} else {
				final long suffixDelta = suffixCoder.decompress(bits);
				newWord = currWord;
				nextSuffix = (currSuffix + suffixDelta);
			}
			currKey = combineToKey(newWord, nextSuffix);
			currWord = newWord;
			currSuffix = nextSuffix;
			final long currOffset = offset + k;
			final boolean foundKey = searchOffset >= 0 ? searchOffset == currOffset : currKey == searchKey;
			compressibleValues.decompress(bits, ngramOrder, !foundKey, outputVal);
			if (foundKey) { return searchOffset >= 0 ? currKey : currOffset; }
			if (searchOffset >= 0) {
				if (currOffset > searchOffset) return -1;
			} else if (currKey > searchKey) return -1;

		}
		return -1;

	}

	/**
	 * @param compressed
	 * @param pos
	 * @return
	 */
	private BitStream getCompressedBits(final LongArray compressed, final long pos) {
		final short bitLength = readShort(compressed.get(pos));
		final BitStream bits = new BitStream(compressed, pos, Short.SIZE, bitLength);
		return bits;
	}

	private short readShort(final long l) {
		return (short) (l >>> (Long.SIZE - Short.SIZE));
	}

	private long decompressSearch(final LongArray compressed, final long searchKey, final int ngramOrder, final T outputVal) {

		return decompressSearch(compressed, searchKey, ngramOrder, outputVal, -1);
	}

	private long decompressSearch(final LongArray compressed, final long searchKey, final int ngramOrder, final T outputVal, final long searchOffset) {

		if (ngramOrder == 0) {
			final boolean lookingForOffset = searchKey >= 0;
			final int word = lookingForOffset ? wordOf(searchKey) : (int) searchOffset;
			if (word < 0 || word >= maps[0].size()) return -1;
			if (outputVal != null) values.getFromOffset(word, 0, outputVal);
			return lookingForOffset ? word : combineToKey(word, 0);
		} else {
			if (compressed == null) return -1;
			final long fromIndex = 0;
			final long toIndex = ((compressed.size() / compressedBlockSize) - 1);
			final long low = binarySearchBlocks(compressed, compressed.size(), searchKey, fromIndex, toIndex, searchOffset);
			if (low < 0) return -1;

			final long index = decompressLinearSearch(compressed, low, searchKey, ngramOrder, outputVal, searchOffset);
			return index;
		}
	}

	/**
	 * @param compressed
	 * @param searchKey
	 * @return
	 */
	private long binarySearchBlocks(final LongArray compressed, final long size, final long searchKey, final long low_, final long high_,
		final long searchOffset) {
		final long toFind = searchOffset >= 0 ? searchOffset : searchKey;
		long low = low_;
		long high = high_;
		assert size % compressedBlockSize == 0;

		while (low <= high) {
			final long mid = (low + high) >>> 1;
			final long currPos = mid * compressedBlockSize;
			final long midVal = searchOffset >= 0 ? offsetCoder.decompress(getCompressedBits(compressed, currPos + 1)) : compressed.get(currPos);
			final int compare = compareLongsRaw(midVal, toFind);
			if (compare < 0) //midVal < key
				low = mid + 1;
			else if (compare > 0) // midVal > key
				high = mid - 1;
			else {
				low = mid + 1;
				break;// key found
			}
		}
		if (low <= 0) return -1;
		final long i = (low - 1) * compressedBlockSize;
		return i;
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
		((CompressibleValueContainer<T>) values).swap(a, b, ngramOrder);
	}

	protected void swap(final LongArray array, final long a, final long b) {
		final long temp = array.get(a);
		array.set(a, array.get(b));
		array.set(b, temp);
	}

	@Override
	public void trim() {
		values.trim();

	}

	@Override
	public void initWithLengths(final List<Long> numNGrams) {
	}

	@Override
	public int getMaxNgramOrder() {
		return maps.length;
	}

	@Override
	public Iterable<Entry<T>> getNgramsForOrder(final int ngramOrder) {
		return new Iterable<Entry<T>>()
		{

			@Override
			public Iterator<edu.berkeley.nlp.lm.map.NgramMap.Entry<T>> iterator() {
				return new Iterator<edu.berkeley.nlp.lm.map.NgramMap.Entry<T>>()
				{

					long currOffset = 0;

					@Override
					public boolean hasNext() {
						return currOffset < maps[ngramOrder].size();
					}

					@Override
					public edu.berkeley.nlp.lm.map.NgramMap.Entry<T> next() {
						final T scratch_ = values.getScratchValue();
						long offset = currOffset;
						final int[] ngram = new int[ngramOrder + 1];
						for (int i = ngramOrder; i >= 0; --i) {
							final T scratch = i == ngramOrder ? scratch_ : null;
							final long foundKey = decompressSearch(maps[i].compressedKeys, -1, i, scratch, offset);
							assert foundKey >= 0;
							ngram[reverseTrie ? (ngramOrder - i) : i] = wordOf(foundKey);
							offset = contextOffsetOf(foundKey);
						}
						currOffset++;

						return new Entry<T>(ngram, scratch_);
					}

					@Override
					public void remove() {
						throw new UnsupportedOperationException("Method not yet implemented");
					}
				};
			}
		};
	}

	@Override
	public long getNumNgrams(final int ngramOrder) {
		return maps[ngramOrder].size();
	}

	@Override
	public boolean contains(final int[] ngram, final int startPos, final int endPos) {
		return getContextOffset(ngram, startPos, endPos, null) >= 0;

	}

	@Override
	public T get(int[] ngram, int startPos, int endPos) {
		T val = values.getScratchValue();
		final long offset = getContextOffset(ngram, startPos, endPos, val);
		if (offset < 0) { return null; }
		return val;

	}

	@Override
	public CustomWidthArray getValueStoringArray(final int ngramOrder) {
		return null;
	}

	@Override
	public void clearStorage() {
		for (int i = 0; i < maps.length; ++i)
			maps[i] = null;
	}

}
