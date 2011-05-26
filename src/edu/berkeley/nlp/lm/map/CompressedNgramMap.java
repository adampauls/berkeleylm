package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.encoding.BitCompressor;
import edu.berkeley.nlp.lm.encoding.VariableLengthBlockCoder;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.values.ValueContainer;

public class CompressedNgramMap<T> extends BinarySearchNgramMap<T> implements Serializable
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

	private long numDecompressQueries = 0;

	private double totalKeyBitsFinal = 0;

	private double totalValueBitsFinal = 0;

	private double totalBitsFinal = 0;

	private double totalSizeFinal = 0;

	private static class CompressedSortedMap extends BinarySearchNgramMap.InternalSortedMap
	{
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		@PrintMemoryCount
		LongArray compressedKeys;

	}

	final int offsetDeltaRadix;

	public CompressedNgramMap(final ValueContainer<T> values, final ConfigOptions opts, LongArray[] numNgramsForEachWord) {
		super(values, opts, true);
		maps = new CompressedSortedMap[numNgramsForEachWord.length];
		offsetCoder = new VariableLengthBlockCoder(OFFSET_RADIX);
		wordCoder = new VariableLengthBlockCoder(WORD_RADIX);
		this.offsetDeltaRadix = opts.offsetDeltaRadix;
		suffixCoder = new VariableLengthBlockCoder(offsetDeltaRadix);
		//		this.buildIndex = opts.buildIndex || opts.storeWordsImplicitly;
		this.compressedBlockSize = opts.compressedBlockSize;

	}

	/**
	 * @param word
	 * @param suffixIndex
	 * @return
	 */
	protected static long getKey(final int word, final long suffixIndex) {
		return (((long) word) << (NUM_SUFFIX_BITS)) | suffixIndex;
	}

	protected static int compareLongsRaw(final long a, final long b) {
		assert a >= 0;
		assert b >= 0;
		if (a > b) return +1;
		if (a < b) return -1;
		if (a == b) return 0;
		throw new RuntimeException();
	}

	/**
	 * @param a
	 * @return
	 */
	protected static long suffixIndex(final long a) {
		return (a & SUFFIX_BIT_MASK);
	}

	/**
	 * @param exactHash
	 * @return
	 */
	protected static long firstWord(final long exactHash) {
		return (exactHash & WORD_BIT_MASK) >>> (NUM_SUFFIX_BITS);
	}

	@Override
	protected long getPrefixOffsetHelp(final int[] ngram, final int startPos, final int endPos) {
		long hasValueSuffixIndex = 0;
		if (endPos > startPos) {
			long lastSuffix = 0L;
			for (int ngramOrder = 0; ngramOrder < endPos - startPos; ++ngramOrder) {
				final int firstWord = reverseTrie ? ngram[endPos - ngramOrder - 1] : ngram[startPos + ngramOrder];
				final long hash = getKey(firstWord, lastSuffix);

				final LongArray compressedKeys = ((CompressedSortedMap) maps[ngramOrder]).compressedKeys;
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
		super.handleNgramsFinished(justFinishedOrder);

		compress(justFinishedOrder - 1);

	}

	private void compress(final int ngramOrder) {
		((CompressedSortedMap) maps[ngramOrder]).compressedKeys = compress(maps[ngramOrder].keys, maps[ngramOrder].keys.size(), ngramOrder);
		values.clearStorageAfterCompression(ngramOrder);
		maps[ngramOrder].keys = null;
	}

	@SuppressWarnings("unused")
	private LongArray compress(final LongArray uncompressed, final long uncompressedSize, final int ngramOrder) {
		Logger.startTrack("Compressing");
		LongArray compressedLongArray = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, uncompressedSize >>> 2);

		long pos = 0;
		long kBits = 0;
		long vBits = 0;
		long currBlock = 0;

		while (pos < uncompressedSize) {
			final BitList currBits = new BitList();
			final long firstKey = uncompressed.get(pos);

			if (currBlock % 1000 == 0) Logger.logs("On block " + currBlock + " starting at pos " + pos);
			currBlock++;

			currBits.addLong(firstKey);
			final BitList offsetBits = offsetCoder.compress(pos);

			final BitList firstValueBits = values.getCompressed(pos, ngramOrder);
			BitList restBits = null;
			BitList headerBits = null;
			long kyBits = 0;
			long valBits = 0;
			long currPos = -1;
			boolean wordBitOn = false;
			OUTER: for (;; wordBitOn = true) {
				kyBits = 0;
				valBits = 0;
				long lastFirstWord = firstWord(firstKey);
				long lastSuffixPart = suffixIndex(firstKey);
				restBits = new BitList();
				headerBits = new BitList();

				headerBits.addAll(offsetBits);
				headerBits.add(wordBitOn);
				headerBits.addAll(firstValueBits);
				currPos = pos + 1;
				final BitList newBits = new BitList();
				for (; currPos < uncompressedSize; ++currPos) {
					final long currKey = uncompressed.get(currPos);
					final long currFirstWord = firstWord(currKey);
					final long currSuffixPart = suffixIndex(currKey);

					final long wordDelta = currFirstWord - lastFirstWord;
					final long suffixDelta = currSuffixPart - lastSuffixPart;
					newBits.clear();
					if (wordDelta > 0 && !wordBitOn) continue OUTER;
					if (wordBitOn) {
						final BitList keyBits = wordCoder.compress(wordDelta);
						newBits.addAll(keyBits);
						if (wordDelta > 0) {
							final BitList suffixBits = suffixCoder.compress(currSuffixPart);//SUFFIX_ABS_RADIX);
							newBits.addAll(suffixBits);
						} else {
							final BitList suffixBits = suffixCoder.compress(suffixDelta);
							newBits.addAll(suffixBits);
						}
					} else {

						final BitList suffixBits = suffixCoder.compress(suffixDelta);
						newBits.addAll(suffixBits);
					}

					kyBits += newBits.size();
					lastFirstWord = currFirstWord;
					final long posDiff = currPos - pos - 1;

					//					if (!skipCompressingVals) {
					final BitList valueBits = values.getCompressed(currPos, ngramOrder);
					valBits += valueBits.size();
					newBits.addAll(valueBits);
					//					}
					//					if (finishedMiniIndexSection) {
					//						lastSuffixPart = lastMiniSuffixPart;
					//
					//						lastFirstWord = lastMiniFirstWord;
					//
					//						newMiniIndexBits.addShort((short) (restBits.size() + newBits.size()));
					//						addedMiniIndex = true;
					//						updateMiniIndexLastTime = true;
					//					} else {
					//						if (!absDeltas || wordDelta > 0)
					lastSuffixPart = currSuffixPart;
					//					}
					final int numTotalBitsSize = Short.SIZE;
					final int lengthSoFar = currBits.size() + numTotalBitsSize + headerBits.size() + restBits.size() + newBits.size();
					if (lengthSoFar >= Long.SIZE * compressedBlockSize) {
						break;
					}

					restBits.addAll(newBits);
				}
				break;
			}

			pos = currPos;
			kBits += kyBits;
			vBits += valBits;
			assert restBits != null;
			assert headerBits != null;
			final int bitLength = restBits.size() + headerBits.size();
			assert bitLength <= Short.MAX_VALUE;
			currBits.addShort((short) bitLength);
			currBits.addAll(headerBits);

			currBits.addAll(restBits);
			assert currBits.size() < Long.SIZE * compressedBlockSize;
			long curr = 0L;
			for (int i = 0; i <= Long.SIZE * compressedBlockSize; ++i) {
				if (i % Long.SIZE == 0 && i > 0) {
					//					if (currSize >= compressedLongArray.length) compressedLongArray = Arrays.copyOf(compressedLongArray, compressedLongArray.length * 3 / 2);
					compressedLongArray.add(curr);
					curr = 0;

				}
				curr = (curr << 1) | ((i >= currBits.size() || !currBits.get(i)) ? 0 : 1);
			}
			assert compressedLongArray.size() % compressedBlockSize == 0;
		}
		compressedLongArray.trim();// = Arrays.copyOf(compressedLongArray, (int) currSize);
		final double keyAvg = (double) kBits / uncompressedSize;
		Logger.logss("Key bits " + keyAvg);
		final double valueAvg = (double) vBits / uncompressedSize;
		Logger.logss("Value bits " + valueAvg);
		final double avg = 64 * (double) compressedLongArray.size() / uncompressedSize;
		Logger.logss("Compressed bits " + avg);
		totalKeyBitsFinal += kBits;
		totalValueBitsFinal += vBits;
		totalBitsFinal += compressedLongArray.size();
		totalSizeFinal += uncompressedSize;
		Logger.logss("Total key bits " + totalKeyBitsFinal / totalSizeFinal);
		Logger.logss("Total value bits " + totalValueBitsFinal / totalSizeFinal);
		Logger.logss("Total bits " + 64.0 * totalBitsFinal / totalSizeFinal);
		//		if (opts.countDeltas) {
		//			String relDeltaFile = Execution.getFile("relDeltas" + ngramOrder + ".out");
		//			writeDeltaFile(relDeltaFile, relDeltas);
		//			String absDeltaFile = Execution.getFile("absDeltas" + ngramOrder + ".out");
		//			writeDeltaFile(absDeltaFile, absDeltasCounter);
		//			String rangesFile = Execution.getFile("ranges" + ngramOrder + ".out");
		//			writeDeltaFile(rangesFile, ranges);
		//
		//		}
		//		compressedLongArray.trim();
		Logger.endTrack();
		return compressedLongArray;
	}

	private long decompressLinearSearch(final LongArray compressed, final long pos, final long searchKey, final int ngramOrder, final T outputVal) {
		numDecompressQueries++;
		final long firstKey = compressed.get(pos);
		//				if (searchKey == 5222680231936L) {
		//					@SuppressWarnings("unused")
		//					int x = 5;
		//				}
		final short bitLength = readShort(compressed.get((pos + 1)));
		final BitStream bits = new BitStream(compressed, pos + 1, Short.SIZE, bitLength);
		final long offset = offsetCoder.decompress(bits);
		final boolean wordBitOn = bits.nextBit();

		long currWord = firstWord(firstKey);
		long currSuffix = suffixIndex(firstKey);
		final boolean foundKeyFirst = firstKey == searchKey;
		//		valueDecompressTimer.start();

		values.decompress(bits, ngramOrder, !foundKeyFirst, outputVal);
		//		valueDecompressTimer.accumStop();
		if (foundKeyFirst) {

		return offset; }
		int k = 1;
		int upperK = Integer.MAX_VALUE;
		//		if (miniIndexActive) {
		//			final long numMiniIndexEntries = bits.next(Byte.SIZE);
		//			miniIndex = new short[(int) numMiniIndexEntries];
		//			for (int i = 0; i < numMiniIndexEntries; ++i) {
		//				final short next = (short) bits.next(Short.SIZE);
		//				assert next >= 0;
		//				miniIndex[i] = next;
		//			}
		//			long currSuffixPart = suffixIndex(firstKey);
		//			long currFirstWord = firstWord(firstKey);
		//			long oldSuffixDelta = -1;
		//			long oldWordDelta = -1;
		//			for (int i = 0; i < numMiniIndexEntries; ++i) {
		//				upperK = (i == numMiniIndexEntries - 1) ? Integer.MAX_VALUE : (((i + 1) * miniIndexNum + 2));
		//				final short currIndexPos = miniIndex[i];
		//
		//				final short lastIndexPos = (i == 0) ? 0 : miniIndex[i - 1];
		//				final int delta = currIndexPos - lastIndexPos;
		//				bits.advance(delta);
		//				if (bits.finished()) {
		//					assert bits.numBitsLeft() == 0;
		//					bits.rewind(delta);
		//					break;
		//				}
		//
		//				bits.mark();
		//				final long wordDelta = wordBitOn ? wordCoder.decompress(bits) : 0;
		//				final long suffixDelta = suffixCoder.decompress(bits);
		//				final long hereKey = joinWordSuffix(currFirstWord + wordDelta, wordDelta > 0 ? suffixDelta : (currSuffixPart + suffixDelta));
		//				final boolean foundKey = hereKey == searchKey;
		//
		//				final T v = skipCompressingVals ? null : values.decompress(bits, ngramOrder, !foundKey);
		//				bits.rewindToMark();
		//
		//				if (foundKey) {
		//					if (valueFound != null) valueFound.value = v;
		//					return offset + miniIndexNum * (i + 1) + 2;
		//				} else if (hereKey > searchKey) {
		//					//must be before this (or not found)
		//					bits.rewind(delta);
		//					break;
		//				} else {
		//					k = miniIndexNum * (i + 1) + 2;
		//					if (oldWordDelta > 0) currWord += oldWordDelta;
		//					if (oldSuffixDelta > 0) {
		//						if (oldWordDelta == 0)
		//							currSuffix += oldSuffixDelta;
		//						else
		//							currSuffix = oldSuffixDelta;
		//					}
		//					oldSuffixDelta = suffixDelta;
		//					oldWordDelta = wordDelta;
		//					if (wordDelta == 0)
		//						currSuffixPart += suffixDelta;
		//					else
		//						currSuffixPart = suffixDelta;
		//					currFirstWord += wordDelta;
		//
		//				}
		//
		//			}
		//		}
		long currKey = -1;

		while (!bits.finished() && k < upperK) {
			long newWord = -1;
			long nextSuffix = -1;

			if (wordBitOn) {

				final long wordDelta = wordCoder.decompress(bits);
				final boolean wordDeltaIsZero = wordDelta == 0;
				final long suffixDelta = suffixCoder.decompress(bits);//wordDeltaIsZero ? suffixRadix : SUFFIX_ABS_RADIX);
				newWord = currWord + wordDelta;
				nextSuffix = wordDeltaIsZero ? (currSuffix + suffixDelta) : suffixDelta;
			} else {
				final long suffixDelta = suffixCoder.decompress(bits);
				newWord = currWord;
				nextSuffix = (currSuffix + suffixDelta);
			}
			currKey = joinWordSuffix(newWord, nextSuffix);

			currWord = newWord;
			currSuffix = nextSuffix;
			//			suffixDecompressTimer.accumStop();
			final boolean foundKey = currKey == searchKey;
			//			valueDecompressTimer.start();
			values.decompress(bits, ngramOrder, !foundKey, outputVal);
			//			valueDecompressTimer.accumStop();
			if (foundKey) {
				final long indexFound = offset + k;
				return indexFound;
			}
			if (currKey > searchKey) {//
				return -1;
			}
			k++;
		}
		return -1;

	}

	private short readShort(final long l) {
		return (short) (l >>> (Long.SIZE - Short.SIZE));
	}

	private long decompressSearch(final LongArray compressed, final long compressedSize, final long searchKey, final int ngramOrder, final T outputVal) {
		//		long searchStart = timersOn ? System.n
		//		int firstWord = (int) firstWord(searchKey);
		final long fromIndex = 0;//!buildIndex ? 0 : maps[ngramOrder].wordRangesLow[firstWord];
		final long toIndex = ((compressedSize / compressedBlockSize) - 1);//!buildIndex ? ((compressedSize / B) - 1) : maps[ngramOrder].wordRangesHigh[firstWord];
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
			final long midVal = compressed.get(mid * compressedBlockSize);//[outerArrayPart(mid)][innerArrayPart(mid)];
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

	//
	//	@Override
	//	public void getValue(final int[] ngram, final int startPos, final int endPos, @OutputParameter final LmContextInfo prefixIndex, @OutputParameter T outputVal) {
	//		assert false : "Untested";
	//		if (containsOutOfVocab(ngram, startPos, endPos)) return;
	//		long lastSuffix = 0L;
	//		for (int ngramOrder = 0; ngramOrder < endPos - startPos; ++ngramOrder) {
	//			final boolean isLast = ngramOrder == endPos - startPos - 1;
	//			final int currStartPos = reverseTrie ? (startPos + ngramOrder + 1) : (startPos);
	//			final int currEndPos = reverseTrie ? (endPos) : (startPos + ngramOrder + 1);
	//			long currIndex = -1;
	//
	//			if (currIndex < 0) {
	//				final long hash = getKey(reverseTrie ? ngram[endPos - ngramOrder - 1] : ngram[startPos + ngramOrder], lastSuffix);
	//				final LongArray compressedKeys = ((CompressedSortedMap) maps[ngramOrder]).compressedKeys;
	//				currIndex = decompressSearch(compressedKeys, compressedKeys.size(), hash, ngramOrder, outputVal);
	//				if (isLast) return;
	//
	//			}
	//			if (currIndex < 0) return;
	//			lastSuffix = currIndex;
	//
	//		}
	//		throw new RuntimeException();
	//
	//	}

	@Override
	public long getValueAndOffset(long contextOffset, int contextNgramOrder, int word, @OutputParameter T outputVal) {

		final long hash = getKey(word, contextOffset);
		final int ngramOrder = contextNgramOrder + 1;
		final LongArray compressedKeys = ((CompressedSortedMap) maps[ngramOrder]).compressedKeys;
		long currIndex = decompressSearch(compressedKeys, compressedKeys.size(), hash, ngramOrder, outputVal);
		return currIndex;

	}

	@Override
	protected BinarySearchNgramMap.InternalSortedMap newInternalSortedMap() {
		return new CompressedSortedMap();
	}

	//	@Override
	//	public LmContextInfo getOffsetForNgram(final int[] phrase, final int startPos, final int endPos) {
	//		throw new UnsupportedOperationException("Method not yet implemented");
	//	}
	//
	//	@Override
	//	public long getOffset(final long prefixIndex, final int prefixNgramOrder, final int word) {
	//		throw new UnsupportedOperationException("Method not yet implemented");
	//	}

	//	@Override
	//	public long getOffset(int[] ngram, int startPos, int endPos) {
	//		// TODO Auto-generated method stub
	//		throw new UnsupportedOperationException("Method not yet implemented");
	//	}

}
