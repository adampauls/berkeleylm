package edu.berkeley.nlp.lm.values;

import java.io.Serializable;
import java.util.Arrays;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitCompressor;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.bits.VariableLengthBitCompressor;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.Logger;

abstract class LmValueContainer<V extends Comparable<V>> implements CompressibleValueContainer<V>, Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	protected LongArray[] valueRanks;

	//@PrintMemoryCount
	//private LongArray[] contextOffsets;

	protected transient Indexer<V> countIndexer;

	protected final boolean storePrefixIndexes;

	protected final BitCompressor valueCoder;

	protected final int valueRadix;

	private final int wordWidth;

	final int rankShift;

	public LmValueContainer(final Indexer<V> countIndexer, final int valueRadix, final boolean storePrefixIndexes) {
		this.valueRadix = valueRadix;
		valueCoder = new VariableLengthBitCompressor(valueRadix);
		this.countIndexer = countIndexer;
		this.storePrefixIndexes = storePrefixIndexes;
		rankShift = this.storePrefixIndexes ? 32 : 0;
		//	if (storePrefixIndexes) contextOffsets = new LongArray[6];
		valueRanks = new LongArray[6];
		countIndexer.getIndex(getDefaultVal());
		countIndexer.trim();
		countIndexer.lock();
		wordWidth = CustomWidthArray.numBitsNeeded(countIndexer.size());
		Logger.startTrack("Storing count indices using " + wordWidth + " bits.");
		storeCounts();
		Logger.endTrack();

	}

	@Override
	public void setMap(NgramMap<V> map) {

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
	public void add(int[] ngram, int startPos, int endPos, final int ngramOrder, final long offset, final long prefixOffset, final int word, final V val_,
		final long suffixOffset, boolean ngramIsNew) {
		V val = val_;
		if (val == null) val = getDefaultVal();

		setSizeAtLeast(10, ngramOrder);

		final int indexOfCounts = countIndexer.getIndex(val);
		if (suffixOffset < 0) {
			@SuppressWarnings("unused")
			int x = 5;
		}
		assert suffixOffset >= 0;
		assert suffixOffset <= Integer.MAX_VALUE;
		if (storePrefixIndexes) {
			valueRanks[ngramOrder].setAndGrowIfNeeded(offset, suffixOffset | (long) indexOfCounts << rankShift);

		} else
			valueRanks[ngramOrder].setAndGrowIfNeeded(offset, indexOfCounts);

	}

	abstract protected V getDefaultVal();

	abstract protected void storeCounts();

	abstract protected void getFromRank(final int rank, @OutputParameter V outputVal);

	@Override
	public void setSizeAtLeast(final long size, final int ngramOrder) {
		if (ngramOrder >= valueRanks.length) {
			valueRanks = Arrays.copyOf(valueRanks, valueRanks.length * 2);
		}
		if (valueRanks[ngramOrder] == null) {
			valueRanks[ngramOrder] = new CustomWidthArray(size, rankShift + wordWidth);
		}
		valueRanks[ngramOrder].ensureCapacity(size + 1);

	}

	public long getSuffixOffset(final long index, final int ngramOrder) {
		return !storePrefixIndexes ? -1 : (int) valueRanks[ngramOrder].get(index);
	}

	@Override
	public void setFromOtherValues(final ValueContainer<V> other) {
		final LmValueContainer<V> o = (LmValueContainer<V>) other;
		this.valueRanks = o.valueRanks;
		this.countIndexer = o.countIndexer;
	}

	protected int getRank(int ngramOrder, long offset) {
		return (int) (valueRanks[ngramOrder].get(offset) >>> rankShift);
	}

	@Override
	public final void decompress(final BitStream bits, final int ngramOrder, final boolean justConsume, @OutputParameter final V outputVal) {
		final long longIndex = doDecode(bits);
		if (justConsume) return;
		final int rank = (int) longIndex;
		if (outputVal != null) getFromRank(rank, outputVal);
	}

	/**
	 * @param bits
	 * @param huffmanEncoder
	 * @return
	 */
	private long doDecode(final BitStream bits) {
		return valueCoder.decompress(bits);
	}

	@Override
	public void clearStorageAfterCompression(final int ngramOrder) {
		if (ngramOrder > 0) valueRanks[ngramOrder] = null;
	}

	@Override
	public BitList getCompressed(final long offset, final int ngramOrder) {
		final int l = getRank(ngramOrder, offset);
		return valueCoder.compress(l);
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {
		valueRanks[ngramOrder].trim();
	}

	@Override
	public void trim() {
		countIndexer = null;

	}

}
