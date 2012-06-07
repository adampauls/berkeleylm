package edu.berkeley.nlp.lm.values;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitCompressor;
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

	//@PrintMemoryCount
	//private LongArray[] contextOffsets;

	protected final boolean storeSuffixIndexes;

	protected final BitCompressor valueCoder;

	protected final int valueRadix;

	protected int wordWidth;

	final int rankShift;
	
	protected final int defaultValRank = 10;

	public RankedValueContainer(final int valueRadix, final boolean storePrefixIndexes, int maxNgramOrder) {
		this.valueRadix = valueRadix;
		valueCoder = new VariableLengthBitCompressor(valueRadix);
		this.storeSuffixIndexes = storePrefixIndexes;
		rankShift = this.storeSuffixIndexes ? Integer.SIZE : 0;
		valueRanks = new CustomWidthArray[maxNgramOrder];

	}

	@Override
	public void setMap(final NgramMap<V> map) {

	}

	@Override
	public void swap(final long a_, final long b_, final int ngramOrder) {

		final int a = (int) a_;
		final int b = (int) b_;
		final long temp = valueRanks[ngramOrder].get(a);
		assert temp >= 0;
		final long val = (int) valueRanks[ngramOrder].get(b);
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

		final int indexOfCounts = getCountRank(val.asLong());
		assert indexOfCounts >= 0;

		if (storeSuffixIndexes) {
			assert suffixOffset >= 0;
			assert suffixOffset <= Integer.MAX_VALUE;
			valueRanks[ngramOrder].setAndGrowIfNeeded(offset, suffixOffset | (long) indexOfCounts << rankShift);
		} else
			valueRanks[ngramOrder].setAndGrowIfNeeded(offset, indexOfCounts);
		return true;

	}

	abstract protected int getCountRank(long val);

	@Override
	public BitList getCompressed(final long offset, final int ngramOrder) {
		final int l = getRank(ngramOrder, offset);
		return valueCoder.compress(l);
	}

	@Override
	public void decompress(final BitStream bits, final int ngramOrder, final boolean justConsume, @OutputParameter final V outputVal) {
		final long longIndex = valueCoder.decompress(bits);
		if (justConsume) return;
		if (outputVal != null)  {final int rank = (int) longIndex;
		getFromRank(rank, outputVal);
		}
	}

	abstract protected V getDefaultVal();

	abstract protected void getFromRank(final int rank, @OutputParameter V outputVal);

	@Override
	public void setSizeAtLeast(final long size, final int ngramOrder) {
		//		if (ngramOrder >= valueRanks.length) {
		//			valueRanks = Arrays.copyOf(valueRanks, valueRanks.length * 2);
		//		}
		if (valueRanks[ngramOrder] == null) {
			//			valueRanks[ngramOrder] = new LongArray(size);
			valueRanks[ngramOrder] = new CustomWidthArray(size, rankShift + wordWidth);
		}
		valueRanks[ngramOrder].ensureCapacity(size + 1);

	}

	public long getSuffixOffset(final long index, final int ngramOrder) {
		return getSuffixOffset(index, valueRanks[ngramOrder]);
	}

	public long getSuffixOffset(final long index, final CustomWidthArray valueRanksForOrder) {
		final long internalVal = valueRanksForOrder.get(index);
		return getSuffixOffsetFromInternalVal(internalVal);
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
		}

	}

	protected int getRank(final int ngramOrder, final long offset) {
		return getRank(valueRanks[ngramOrder], offset);
	}

	protected int getRank(final CustomWidthArray valueRanksForOrder, final long offset) {
		final long internalVal = valueRanksForOrder.get(offset);
		return getRankFromInternalVal(internalVal);
	}

	/**
	 * @param internalVal
	 * @return
	 */
	protected int getRankFromInternalVal(final long internalVal) {
		return (int) (internalVal >>> rankShift);
	}

	@Override
	public void clearStorageAfterCompression(final int ngramOrder) {
		if (ngramOrder > 0) valueRanks[ngramOrder] = null;
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {
		valueRanks[ngramOrder].trim();
	}

	@Override
	public void trim() {

	}

	@Override
	public void clearStorageForOrder(final int ngramOrder) {
		valueRanks[ngramOrder] = null;
	}

	@Override
	public boolean storeSuffixoffsets() {
		return storeSuffixIndexes;
	}

}
