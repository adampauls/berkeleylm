package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.encoding.BitCompressor;
import edu.berkeley.nlp.lm.encoding.VariableLengthBlockCoder;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.values.ValueContainer;

public class CompressedNgramMap<T> extends AbstractNgramMap<T> implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final int compressedBlockSize;

	private static final int OFFSET_RADIX = 33;

	private static final int WORD_RADIX = 2;

	private final BitCompressor offsetCoder;

	private final BitCompressor wordCoder;

	private final BitCompressor suffixCoder;

	private double totalKeyBitsFinal = 0;

	private double totalValueBitsFinal = 0;

	private double totalBitsFinal = 0;

	private double totalSizeFinal = 0;

	private final int offsetDeltaRadix;

	private CompressedMap[] maps;

	private final boolean reverseTrie = true;

	public CompressedNgramMap(final ValueContainer<T> values, final ConfigOptions opts) {
		super(values, opts);
		offsetCoder = new VariableLengthBlockCoder(OFFSET_RADIX);
		wordCoder = new VariableLengthBlockCoder(WORD_RADIX);
		this.offsetDeltaRadix = opts.offsetDeltaRadix;
		suffixCoder = new VariableLengthBlockCoder(offsetDeltaRadix);
		this.compressedBlockSize = opts.compressedBlockSize;
		values.setMap(this);

	}

	@Override
	public long getValueAndOffset(final long contextOffset, final int contextNgramOrder, final int word, @OutputParameter final T outputVal) {

		final long hash = combineToKey(word, contextOffset);
		final int ngramOrder = contextNgramOrder + 1;
		final LongArray compressedKeys = (maps[ngramOrder]).compressedKeys;
		final long currIndex = decompressSearch(compressedKeys, compressedKeys.size(), hash, ngramOrder, outputVal);
		return currIndex;

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.mt.lm.NgramMap#add(java.util.List, T)
	 */
	@Override
	public long put(final int[] ngram, int startPos, int endPos, final T val) {

		final int ngramOrder = endPos - startPos - 1;
		final int word = reverseTrie ? ngram[startPos] : ngram[endPos - 1];

		final long contextOffset = reverseTrie ? getContextOffset(ngram, startPos + 1, endPos) : getContextOffset(ngram, startPos, endPos - 1);
		if (contextOffset < 0) return -1;
		final CompressedMap map = maps[ngramOrder];
		long oldSize = map.size();
		final long newOffset = map.add(combineToKey(word, contextOffset));
		values.add(ngram,startPos,endPos, ngramOrder, map.size() - 1, contextOffset, word, val, (-1), map.size() == oldSize);

		return newOffset;

	}

	private long getContextOffset(final int[] ngram, final int startPos, final int endPos) {
		if (endPos == startPos) return 0;
		long hasValueSuffixIndex = 0;
		if (endPos > startPos) {
			long lastSuffix = 0L;
			for (int ngramOrder = 0; ngramOrder < endPos - startPos; ++ngramOrder) {
				final int firstWord = reverseTrie ? ngram[endPos - ngramOrder - 1] : ngram[startPos + ngramOrder];
				final long hash = combineToKey(firstWord, lastSuffix);

				final LongArray compressedKeys = (maps[ngramOrder]).compressedKeys;
				final long currIndex = decompressSearch(compressedKeys, compressedKeys.size(), hash, ngramOrder, null);
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
		final LongArray currKeys = maps[justFinishedOrder - 1].keys;
		final long currSize = currKeys.size();
		sort(currKeys, 0, currSize - 1, justFinishedOrder - 1);
		maps[justFinishedOrder - 1].trim();
		values.trimAfterNgram(justFinishedOrder - 1, currSize);
		compress(justFinishedOrder - 1);
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
		(maps[ngramOrder]).compressedKeys = compress(maps[ngramOrder].keys, maps[ngramOrder].keys.size(), ngramOrder);
		values.clearStorageAfterCompression(ngramOrder);
		maps[ngramOrder].keys = null;
	}

	private LongArray compress(final LongArray uncompressed, final long uncompressedSize, final int ngramOrder) {
		Logger.startTrack("Compressing");
		final LongArray compressedLongArray = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, uncompressedSize >>> 2);

		long uncompressedPos = 0;
		long totalNumKeyBits = 0;
		long totalNumValueBits = 0;
		long currBlock = 0;

		while (uncompressedPos < uncompressedSize) {
			final BitList currBlockBits = new BitList();
			final long firstKey = uncompressed.get(uncompressedPos);

			if (currBlock++ % 1000 == 0) Logger.logs("On block " + currBlock + " starting at pos " + uncompressedPos);

			currBlockBits.addLong(firstKey);
			final BitList offsetBits = offsetCoder.compress(uncompressedPos);

			final BitList firstValueBits = values.getCompressed(uncompressedPos, ngramOrder);
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
		final BitList valueBits = values.getCompressed(currPos, ngramOrder);
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

	private long decompressLinearSearch(final LongArray compressed, final long pos, final long searchKey, final int ngramOrder, final T outputVal) {
		final long firstKey = compressed.get(pos);
		final short bitLength = readShort(compressed.get((pos + 1)));
		final BitStream bits = new BitStream(compressed, pos + 1, Short.SIZE, bitLength);
		final long offset = offsetCoder.decompress(bits);
		final boolean wordBitOn = bits.nextBit();

		int currWord = wordOf(firstKey);
		long currSuffix = contextOffsetOf(firstKey);
		final boolean foundKeyFirst = firstKey == searchKey;

		values.decompress(bits, ngramOrder, !foundKeyFirst, outputVal);
		if (foundKeyFirst) return offset;

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
			final boolean foundKey = currKey == searchKey;
			values.decompress(bits, ngramOrder, !foundKey, outputVal);
			if (foundKey) {
				final long indexFound = offset + k;
				return indexFound;
			}
			if (currKey > searchKey) return -1;

		}
		return -1;

	}

	private short readShort(final long l) {
		return (short) (l >>> (Long.SIZE - Short.SIZE));
	}

	private long decompressSearch(final LongArray compressed, final long compressedSize, final long searchKey, final int ngramOrder, final T outputVal) {
		final long fromIndex = 0;
		final long toIndex = ((compressedSize / compressedBlockSize) - 1);
		final long low = binarySearchBlocks(compressed, compressedSize, searchKey, fromIndex, toIndex);
		if (low < 0) return -1;

		final long index = decompressLinearSearch(compressed, low, searchKey, ngramOrder, outputVal);
		return index;
	}

	/**
	 * @param compressed
	 * @param searchKey
	 * @return
	 */
	private long binarySearchBlocks(final LongArray compressed, final long size, final long searchKey, final long low_, final long high_) {
		long low = low_;
		long high = high_;
		assert size % compressedBlockSize == 0;

		while (low <= high) {
			final long mid = (low + high) >>> 1;
			final long midVal = compressed.get(mid * compressedBlockSize);
			final int compare = compareLongsRaw(midVal, searchKey);
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
		values.swap(a, b, ngramOrder);
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
		maps = new CompressedMap[numNGrams.size()];
		for (int i = 0; i < numNGrams.size(); ++i) {
			maps[i] = new CompressedMap();
			final long l = numNGrams.get(i);
			maps[i].init(l);
			values.setSizeAtLeast(l, i);

		}
	}

}
