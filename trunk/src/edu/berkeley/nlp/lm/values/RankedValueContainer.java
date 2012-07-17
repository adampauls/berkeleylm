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

	protected final boolean storeSuffixIndexes;

	protected final VariableLengthBitCompressor valueCoder;

	protected final int valueRadix;

	protected int valueWidth;

	protected final int defaultValRank = 10;

	protected final long[] numNgramsForEachOrder;

	protected final int[] suffixBitsForOrder;

	protected boolean useMapValueArray = false;

	private NgramMap<V> ngramMap;

	public RankedValueContainer(final int valueRadix, final boolean storePrefixIndexes, long[] numNgramsForEachOrder) {
		this.valueRadix = valueRadix;
		suffixBitsForOrder = new int[numNgramsForEachOrder.length];

		this.numNgramsForEachOrder = numNgramsForEachOrder;
		valueCoder = new VariableLengthBitCompressor(valueRadix);
		this.storeSuffixIndexes = storePrefixIndexes;
		valueRanks = new CustomWidthArray[numNgramsForEachOrder.length];

	}

	@Override
	public void setMap(final NgramMap<V> map) {
		this.ngramMap = map;

	}

	protected boolean useValueStoringArray() {
		return false;
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

		assert suffixOffset < 0 || ngramOrder == 0 || CustomWidthArray.numBitsNeeded(suffixOffset) <= suffixBitsForOrder[ngramOrder] : "Problem with suffix offset bits "
			+ suffixOffset + " " + numNgramsForEachOrder[ngramOrder - 1] + " " + Arrays.toString(ngram);
		V val = val_;
		if (val == null) val = getDefaultVal();

		setSizeAtLeast(10, ngramOrder);

		final long indexOfCounts = getCountRank(val.asLong());

		assert indexOfCounts >= 0;

		final CustomWidthArray valueRanksHere = valueRanks[ngramOrder];
		final int widthOffset = ngramOrder == 0 || !useMapValueArray ? 0 : valueRanksHere.getKeyWidth();
		valueRanksHere.setAndGrowIfNeeded(offset, indexOfCounts, widthOffset, valueWidth);
		if (storeSuffixIndexes && ngramOrder > 0) {
			assert suffixOffset >= 0;
			assert suffixOffset <= Integer.MAX_VALUE;
			valueRanksHere.setAndGrowIfNeeded(offset, suffixOffset, widthOffset + valueWidth, suffixBitsForOrder[ngramOrder]);
		}
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
			final int suffixBits = (ngramOrder == 0) ? 0 : suffixBitsForOrder[ngramOrder];

			if (ngramOrder < suffixBitsForOrder.length - 1)
				suffixBitsForOrder[ngramOrder + 1] = !storeSuffixIndexes ? 0 : CustomWidthArray.numBitsNeeded(size);
			final CustomWidthArray valueStoringArray = ngramMap.getValueStoringArray(ngramOrder);
			final boolean useValueStoringArrayHere = valueStoringArray != null && useValueStoringArray();
			if (useValueStoringArrayHere) {
				useMapValueArray = true;
				valueRanks[ngramOrder] = valueStoringArray;
			} else {
				valueRanks[ngramOrder] = new CustomWidthArray(size, valueWidth, valueWidth + suffixBits);
				valueRanks[ngramOrder].setAndGrowIfNeeded(size - 1, 0L);
			}
		}
	}

	public long getSuffixOffset(final long index, final int ngramOrder) {
		assert ngramOrder > 0;
		final CustomWidthArray valueRanksHere = valueRanks[ngramOrder];
		final int widthOffset = !useMapValueArray ? 0 : valueRanksHere.getKeyWidth();
		final int width = widthOffset + valueWidth;
		return valueRanksHere.get(index, width, valueRanksHere.getFullWidth() - width);
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

	protected long getRank(final int ngramOrder, final long offset) {
		final CustomWidthArray valueRanksHere = valueRanks[ngramOrder];
		final int widthOffset = ngramOrder == 0 || !useMapValueArray ? 0 : valueRanksHere.getKeyWidth();
		return valueRanksHere.get(offset, widthOffset, valueWidth);
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

	@Override
	public int numValueBits(int ngramOrder) {
		return valueWidth + suffixBitsForOrder[ngramOrder];
	}

}
