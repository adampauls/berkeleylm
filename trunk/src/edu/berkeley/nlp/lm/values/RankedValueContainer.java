package edu.berkeley.nlp.lm.values;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.bits.VariableLengthBitCompressor;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry;
import edu.berkeley.nlp.lm.collections.LongRepresentable;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;

abstract class RankedValueContainer<V extends LongRepresentable<V>> implements CompressibleValueContainer<V>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	protected final CustomWidthArray[] valueRanks;

	@PrintMemoryCount
	protected final CustomWidthArray[] suffixOffsets;

	protected final boolean storeSuffixIndexes;

	protected final VariableLengthBitCompressor valueCoder;

	protected final int valueRadix;

	protected int wordWidth;

	protected final int defaultValRank = 10;

	protected final long[] numNgramsForEachOrder;

	public RankedValueContainer(final int valueRadix, final boolean storePrefixIndexes, long[] numNgramsForEachOrder) {
		this.valueRadix = valueRadix;
		suffixOffsets = storePrefixIndexes ? new CustomWidthArray[numNgramsForEachOrder.length] : null;
		this.numNgramsForEachOrder = numNgramsForEachOrder;
		valueCoder = new VariableLengthBitCompressor(valueRadix);
		this.storeSuffixIndexes = storePrefixIndexes;
		if (storeSuffixIndexes) {
			for (int i = 1; i < numNgramsForEachOrder.length; ++i) {
				suffixOffsets[i] = new CustomWidthArray(numNgramsForEachOrder[i], CustomWidthArray.numBitsNeeded(numNgramsForEachOrder[i - 1]));
			}
		}
		valueRanks = new CustomWidthArray[numNgramsForEachOrder.length];

	}

	@Override
	public void setMap(final NgramMap<V> map) {

	}

	@Override
	public void swap(final long a, final long b, final int ngramOrder) {

		final long temp = valueRanks[ngramOrder].get(a);
		assert temp >= 0;
		final long val = valueRanks[ngramOrder].get(b);
		assert val >= 0;
		valueRanks[ngramOrder].set(a, val);
		valueRanks[ngramOrder].set(b, temp);
	}

	@Override
	public boolean add(final int[] ngram, final int startPos, final int endPos, final int ngramOrder, final long offset, final long prefixOffset,
		final int word, final V val_, final long suffixOffset, final boolean ngramIsNew) {
		if (suffixOffset < 0 && storeSuffixIndexes) return false;
		V val = val_;
		if (val == null) val = getDefaultVal();

		setSizeAtLeast(10, ngramOrder);

		final long indexOfCounts = getCountRank(val.asLong());

		assert indexOfCounts >= 0;

		if (storeSuffixIndexes && ngramOrder > 0) {
			assert suffixOffset >= 0;
			assert suffixOffset <= Integer.MAX_VALUE;
			suffixOffsets[ngramOrder].setAndGrowIfNeeded(offset, suffixOffset);
		}
		valueRanks[ngramOrder].setAndGrowIfNeeded(offset, indexOfCounts);
		return true;

	}

	abstract protected long getCountRank(long val);

	@Override
	public BitList getCompressed(final long offset, final int ngramOrder) {
		final long l = getRank(ngramOrder, offset);
		return valueCoder.compress(l);
	}

	@Override
	public void decompress(final BitStream bits, final int ngramOrder, final boolean justConsume, @OutputParameter final V outputVal) {
		final long longIndex = valueCoder.decompress(bits);
		if (justConsume) return;
		if (outputVal != null) {
			final int rank = (int) longIndex;
			getFromRank(rank, outputVal);
		}
	}

	abstract protected V getDefaultVal();

	abstract protected void getFromRank(final long rank, @OutputParameter V outputVal);

	@Override
	public void setSizeAtLeast(final long size, final int ngramOrder) {
		if (valueRanks[ngramOrder] == null) {
			valueRanks[ngramOrder] = new CustomWidthArray(size, wordWidth);
		}
		valueRanks[ngramOrder].ensureCapacity(size + 1);

	}

	public long getSuffixOffset(final long index, final int ngramOrder) {
		return getSuffixOffset(index, suffixOffsets[ngramOrder]);
	}

	public long getSuffixOffset(final long index, final CustomWidthArray suffixOffsetsForOrder) {
		return getRank(suffixOffsetsForOrder, index);
	}

	/**
	 * @param internalVal
	 * @return
	 */
	protected int getSuffixOffsetFromInternalVal(final long internalVal) {
		return !storeSuffixIndexes ? -1 : (int) internalVal;
	}

	@Override
	public void setFromOtherValues(final ValueContainer<V> other) {
		final RankedValueContainer<V> o = (RankedValueContainer<V>) other;
		for (int i = 0; i < valueRanks.length; ++i) {
			this.valueRanks[i] = o.valueRanks[i];
			this.suffixOffsets[i] = o.suffixOffsets[i];
		}

	}

	protected long getRank(final int ngramOrder, final long offset) {
		final CustomWidthArray valueRanksHere = valueRanks[ngramOrder];
		return getRank(valueRanksHere, offset);
	}

	/**
	 * @param offset
	 * @param customWidthArray
	 * @return
	 */
	protected long getRank(final CustomWidthArray customWidthArray, final long offset) {
		return customWidthArray.get(offset);
	}

	@Override
	public void clearStorageAfterCompression(final int ngramOrder) {
		if (ngramOrder > 0) valueRanks[ngramOrder] = null;
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {
		valueRanks[ngramOrder].trim();
		if (suffixOffsets != null && suffixOffsets[ngramOrder] != null) suffixOffsets[ngramOrder].trim();
	}

	@Override
	public void trim() {

	}

	@Override
	public void clearStorageForOrder(final int ngramOrder) {
		valueRanks[ngramOrder] = null;
		if (suffixOffsets != null) suffixOffsets[ngramOrder] = null;
	}

	@Override
	public boolean storeSuffixoffsets() {
		return storeSuffixIndexes;
	}

}
