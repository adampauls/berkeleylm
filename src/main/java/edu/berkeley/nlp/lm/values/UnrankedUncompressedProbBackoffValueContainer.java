package edu.berkeley.nlp.lm.values;

import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.bits.BitUtils;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

public final class UnrankedUncompressedProbBackoffValueContainer implements ProbBackoffValueContainer
{

	private static final long serialVersionUID = 964277160049236607L;

	private final boolean storeSuffixIndexes;

	private final int[] suffixBitsForOrder;

	private final long[] numNgramsForEachOrder;

	private CustomWidthArray[] valueRanks = null;

	private NgramMap<ProbBackoffPair> ngramMap;

	public UnrankedUncompressedProbBackoffValueContainer(final boolean storePrefixes, long[] numNgramsForEachOrder) {
		this.storeSuffixIndexes = storePrefixes;
		this.numNgramsForEachOrder = numNgramsForEachOrder;
		this.valueRanks = new CustomWidthArray[numNgramsForEachOrder.length];
		suffixBitsForOrder = new int[numNgramsForEachOrder.length];
	}

	@Override
	public UnrankedUncompressedProbBackoffValueContainer createFreshValues(long[] numNgramsForEachOrder_) {
		return new UnrankedUncompressedProbBackoffValueContainer(storeSuffixIndexes, numNgramsForEachOrder_);
	}

	@Override
	public final float getProb(final int ngramOrder, final long index) {
		return ProbBackoffPair.probOf(getProbBackoff(ngramOrder, index));
	}

	/**
	 * @param ngramOrder
	 * @param index
	 * @return
	 */
	private long getProbBackoff(final int ngramOrder, final long index) {
		return valueRanks[ngramOrder].get(index, ngramOrder == 0 ? 0 : valueRanks[ngramOrder].getKeyWidth(), numProbBackoffBits(ngramOrder));
	}

	@Override
	public void getFromOffset(final long index, final int ngramOrder, @OutputParameter final ProbBackoffPair outputVal) {
		long l = getProbBackoff(ngramOrder, index);
		outputVal.prob = ProbBackoffPair.probOf(l);
		outputVal.backoff = ProbBackoffPair.backoffOf(l);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.values.IProb#getBackoff(int, long)
	 */
	@Override
	public final float getBackoff(final int ngramOrder, final long index) {
		return ProbBackoffPair.backoffOf(getProbBackoff(ngramOrder, index));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.values.IProb#getScratchValue()
	 */
	@Override
	public ProbBackoffPair getScratchValue() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}

	@Override
	public void setFromOtherValues(final ValueContainer<ProbBackoffPair> other) {
		final UnrankedUncompressedProbBackoffValueContainer o = (UnrankedUncompressedProbBackoffValueContainer) other;
		for (int i = 0; i < valueRanks.length; ++i) {
			this.valueRanks[i] = o.valueRanks[i];
		}
	}

	@Override
	public void trim() {
	}

	@Override
	public boolean storeSuffixoffsets() {
		return storeSuffixIndexes;
	}

	@Override
	public int numValueBits(int ngramOrder) {
		return numProbBackoffBits(ngramOrder) + suffixBitsForOrder[ngramOrder];
	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private int numProbBackoffBits(int ngramOrder) {
		return (ngramOrder == numNgramsForEachOrder.length - 1 ? Float.SIZE : 2 * Float.SIZE);
	}

	@Override
	public boolean add(int[] ngram, int startPos, int endPos, int ngramOrder, long offset, long contextOffset, int word, ProbBackoffPair val_,
		long suffixOffset, boolean ngramIsNew) {
		if (suffixOffset < 0 && storeSuffixIndexes) return false;

		assert suffixOffset < 0 || ngramOrder == 0 || CustomWidthArray.numBitsNeeded(suffixOffset) <= suffixBitsForOrder[ngramOrder] : "Problem with suffix offset bits "
			+ suffixOffset + " " + numNgramsForEachOrder[ngramOrder - 1] + " " + Arrays.toString(ngram);
		ProbBackoffPair val = val_;
		if (val == null) val = getScratchValue();

		setSizeAtLeast(10, ngramOrder);

		final long indexOfCounts = val.asLong();

		final CustomWidthArray valueRanksHere = valueRanks[ngramOrder];
		final int widthOffset = ngramOrder == 0 ? 0 : valueRanksHere.getKeyWidth();
		valueRanksHere.setAndGrowIfNeeded(offset, ngramOrder == valueRanks.length - 1 ? BitUtils.getLowLong(indexOfCounts) : indexOfCounts, widthOffset,
			numProbBackoffBits(ngramOrder));
		if (storeSuffixIndexes && ngramOrder > 0) {
			assert suffixOffset >= 0;
			assert suffixOffset <= Integer.MAX_VALUE;
			valueRanksHere.setAndGrowIfNeeded(offset, suffixOffset, widthOffset + numProbBackoffBits(ngramOrder), suffixBitsForOrder[ngramOrder]);
		}
		return true;
	}

	@Override
	public void setSizeAtLeast(long size, int ngramOrder) {
		if (valueRanks[ngramOrder] == null) {
			final int suffixBits = (ngramOrder == 0 || !storeSuffixIndexes) ? 0 : suffixBitsForOrder[ngramOrder];

			if (storeSuffixIndexes && ngramOrder < suffixBitsForOrder.length - 1) suffixBitsForOrder[ngramOrder + 1] = CustomWidthArray.numBitsNeeded(size);

			final CustomWidthArray valueStoringArray = ngramMap.getValueStoringArray(ngramOrder);
			final boolean useValueStoringArrayHere = valueStoringArray != null;
			if (useValueStoringArrayHere) {
				valueRanks[ngramOrder] = valueStoringArray;
			} else {
				valueRanks[ngramOrder] = new CustomWidthArray(size, numProbBackoffBits(ngramOrder) + suffixBits);
				valueRanks[ngramOrder].setAndGrowIfNeeded(size - 1, getScratchValue().asLong());
			}
		}
	}

	@Override
	public long getSuffixOffset(final long index, final int ngramOrder) {
		assert ngramOrder > 0;
		final CustomWidthArray valueRanksHere = valueRanks[ngramOrder];
		final int widthOffset = valueRanksHere.getKeyWidth();
		final int width = widthOffset + numProbBackoffBits(ngramOrder);
		return valueRanksHere.get(index, width, valueRanksHere.getFullWidth() - width);
	}

	@Override
	public void trimAfterNgram(int ngramOrder, long size) {
	}

	@Override
	public void setMap(NgramMap<ProbBackoffPair> map) {
		this.ngramMap = map;
	}

	@Override
	public void clearStorageForOrder(int ngramOrder) {
	}
}