package edu.berkeley.nlp.lm.values;

import java.util.List;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

public final class UncompressedProbBackoffValueContainer extends RankedValueContainer<ProbBackoffPair> implements ProbBackoffValueContainer
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	final long[] probsAndBackoffsForRank; // ugly: we encode probs and backoffs consecutively in this area to improve cache locality

	transient LongToIntHashMap countIndexer;

	public UncompressedProbBackoffValueContainer(final LongToIntHashMap countCounter, final int valueRadix, final boolean storePrefixes,
		long[] numNgramsForEachOrder) {
		super(valueRadix, storePrefixes, numNgramsForEachOrder);
		Logger.startTrack("Storing values");
		final long defaultVal = getDefaultVal().asLong();
		final boolean hasDefaultVal = countCounter.get(defaultVal, -1) >= 0;
		probsAndBackoffsForRank = new long[(countCounter.size() + (hasDefaultVal ? 0 : 1))];
		countIndexer = new LongToIntHashMap();
		int k = 0;
		for (final Entry pair : countCounter.getObjectsSortedByValue(true)) {

			countIndexer.put(pair.key, countIndexer.size());
			probsAndBackoffsForRank[k++] = pair.key;
			if (countIndexer.size() == defaultValRank && !hasDefaultVal) {
				countIndexer.put(defaultVal, countIndexer.size());
				probsAndBackoffsForRank[k++] = defaultVal;

			}
		}
		if (countIndexer.size() < defaultValRank && !hasDefaultVal) {
			countIndexer.put(defaultVal, countIndexer.size());
			probsAndBackoffsForRank[k++] = defaultVal;
		}
		valueWidth = CustomWidthArray.numBitsNeeded(countIndexer.size());
		Logger.logss("Storing count indices using " + valueWidth + " bits.");
		Logger.endTrack();
	}

	/**
	 * @param valueRadix
	 * @param storePrefixIndexes
	 * @param maxNgramOrder
	 * @param hasBackoffValIndexer
	 * @param noBackoffValIndexer
	 * @param probsAndBackoffsForRank
	 * @param probsForRank
	 * @param hasBackoffValIndexer
	 */
	public UncompressedProbBackoffValueContainer(int valueRadix, boolean storePrefixIndexes, long[] numNgramsForEachOrder, long[] probsAndBackoffsForRank,
		LongToIntHashMap countIndexer, int wordWidth) {
		super(valueRadix, storePrefixIndexes, numNgramsForEachOrder);
		this.countIndexer = countIndexer;
		this.probsAndBackoffsForRank = probsAndBackoffsForRank;
		super.valueWidth = wordWidth;
	}

	@Override
	public UncompressedProbBackoffValueContainer createFreshValues(long[] numNgramsForEachOrder_) {
		return new UncompressedProbBackoffValueContainer(valueRadix, storeSuffixIndexes, numNgramsForEachOrder_, probsAndBackoffsForRank, countIndexer,
			valueWidth);
	}

	@Override
	public final float getProb(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, false);
	}

	public final long getInternalVal(final int ngramOrder, final long index) {
		return valueRanks[ngramOrder].get(index);
	}

	public final float getProb(final CustomWidthArray valueRanksForOrder, final long index) {
		return getCount(valueRanksForOrder, index, false);
	}

	@Override
	public void getFromOffset(final long index, final int ngramOrder, @OutputParameter final ProbBackoffPair outputVal) {
		final long rank = getRank(ngramOrder, index);
		getFromRank(rank, outputVal);
	}

	/**
	 * @param ngramOrder
	 * @param index
	 * @param uncompressProbs2
	 * @return
	 */
	private float getCount(final int ngramOrder, final long index, final boolean backoff) {
		final long rank = getRank(ngramOrder, index);
		return getFromRank(rank, backoff);
	}

	private float getCount(final CustomWidthArray valueRanksForOrder, final long index, final boolean backoff) {
		final long rank = valueRanksForOrder.get(index);
		return getFromRank(rank, backoff);
	}

	private float getFromRank(final long rank, final boolean backoff) {
		return backoff ? ProbBackoffPair.backoffOf(probsAndBackoffsForRank[(int) rank]) : ProbBackoffPair.probOf(probsAndBackoffsForRank[(int) rank]);//backoff ? backoffsForRank[backoffRankOf(val)] : probsForRank[probRankOf(val)];
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see edu.berkeley.nlp.lm.values.IProb#getBackoff(int, long)
	 */
	@Override
	public final float getBackoff(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, true);
	}

	public final float getBackoff(final CustomWidthArray valueRanksForNgramOrder, final long index) {
		return getCount(valueRanksForNgramOrder, index, true);
	}

	@Override
	protected ProbBackoffPair getDefaultVal() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}

	@Override
	protected void getFromRank(final long rank, @OutputParameter final ProbBackoffPair outputVal) {

		outputVal.prob = getFromRank(rank, false);
		outputVal.backoff = getFromRank(rank, true);
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
	public void setFromOtherValues(final ValueContainer<ProbBackoffPair> o) {
		super.setFromOtherValues(o);
		this.countIndexer = ((UncompressedProbBackoffValueContainer) o).countIndexer;
	}

	@Override
	public void trim() {
		super.trim();
		countIndexer = null;
	}

	@Override
	protected long getCountRank(long val) {
		return countIndexer.get(val, -1);
	}

	@Override
	protected boolean useValueStoringArray() {
		return true;
	}

}