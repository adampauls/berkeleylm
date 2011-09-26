package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

public final class ProbBackoffValueContainer extends RankedValueContainer<ProbBackoffPair>
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	final float[] probsAndBackoffsForRank; // ugly, but we but probs and backoffs consecutively in this area to improve cache locality

	public ProbBackoffValueContainer(final Indexer<ProbBackoffPair> countIndexer, final int valueRadix, final boolean storePrefixes, int maxNgramOrder) {
		super(countIndexer, valueRadix, storePrefixes, maxNgramOrder);
		Logger.startTrack("Storing count indices using " + wordWidth + " bits.");
		probsAndBackoffsForRank = new float[2 * this.countIndexer.size()];
		int k = 0;
		for (final ProbBackoffPair pair : this.countIndexer.getObjects()) {

			probsAndBackoffsForRank[k++] = pair.prob;
			probsAndBackoffsForRank[k++] = pair.backoff;
		}
		Logger.endTrack();
	}

	@Override
	public ProbBackoffValueContainer createFreshValues() {
		return new ProbBackoffValueContainer(countIndexer, valueRadix, storeSuffixIndexes, valueRanks.length);
	}

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
		final int rank = getRank(ngramOrder, index);
		getFromRank(rank, outputVal);
	}

	/**
	 * @param ngramOrder
	 * @param index
	 * @param uncompressProbs2
	 * @return
	 */
	private float getCount(final int ngramOrder, final long index, final boolean backoff) {
		final int rank = getRank(ngramOrder, index);
		return getFromRank(rank, backoff);
	}

	private float getCount(final CustomWidthArray valueRanksForOrder, final long index, final boolean backoff) {
		final int rank = getRank(valueRanksForOrder, index);
		return getFromRank(rank, backoff);
	}

	private float getFromRank(final int rank, final boolean backoff) {
		return probsAndBackoffsForRank[2 * rank + (backoff ? 1 : 0)];
	}

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
	protected void getFromRank(final int rank, @OutputParameter final ProbBackoffPair outputVal) {

		outputVal.prob = getFromRank(rank, false);
		outputVal.backoff = getFromRank(rank, true);
	}

	@Override
	public ProbBackoffPair getScratchValue() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}



}