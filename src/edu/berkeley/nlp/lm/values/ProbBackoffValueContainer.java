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

public final class ProbBackoffValueContainer extends RankedValueContainer<ProbBackoffPair>
{

	private static final long serialVersionUID = 964277160049236607L;

	private transient LongToIntHashMap hasBackoffValIndexer;

	private transient LongToIntHashMap noBackoffValIndexer;

	private static final int HAS_BACKOFF = 1;

	private static final int NO_BACKOFF = 0;

	@PrintMemoryCount
	final long[] probsAndBackoffsForRank; // ugly, but we but probs and backoffs consecutively in this area to improve cache locality

	@PrintMemoryCount
	final float[] probsForRank; // ugly, but we but probs and backoffs consecutively in this area to improve cache locality

	public ProbBackoffValueContainer(final LongToIntHashMap countCounter, final int valueRadix, final boolean storePrefixes, int maxNgramOrder) {
		super(valueRadix, storePrefixes, maxNgramOrder);
		Logger.startTrack("Storing values");
		final boolean hasDefaultVal = countCounter.get(getDefaultVal().asLong(), -1) >= 0;
		hasBackoffValIndexer = new LongToIntHashMap();
		noBackoffValIndexer = new LongToIntHashMap();
		List<Entry> objectsSortedByValue = countCounter.getObjectsSortedByValue(true);
		for (Entry e : objectsSortedByValue) {
			if (ProbBackoffPair.backoffOf(e.key) == 0.0f) {
				noBackoffValIndexer.put(e.key, noBackoffValIndexer.size());
			} else {
				hasBackoffValIndexer.put(e.key, hasBackoffValIndexer.size());
				if (hasBackoffValIndexer.size() == defaultValRank && !hasDefaultVal) {
					hasBackoffValIndexer.put(getDefaultVal().asLong(), hasBackoffValIndexer.size());

				}
			}
		}
		if (hasBackoffValIndexer.size() < defaultValRank && !hasDefaultVal) {
			hasBackoffValIndexer.put(getDefaultVal().asLong(), hasBackoffValIndexer.size());

		}
		probsAndBackoffsForRank = new long[hasBackoffValIndexer.size()];
		probsForRank = new float[noBackoffValIndexer.size()];
		wordWidth = CustomWidthArray.numBitsNeeded(Math.max(probsAndBackoffsForRank.length, probsForRank.length)) + 1;
		for (java.util.Map.Entry<Long, Integer> entry : hasBackoffValIndexer.entries()) {
			probsAndBackoffsForRank[entry.getValue()] = entry.getKey();
		}
		for (java.util.Map.Entry<Long, Integer> entry : noBackoffValIndexer.entries()) {
			probsForRank[(entry.getValue())] = ProbBackoffPair.probOf(entry.getKey());
		}

		Logger.logss("Storing count indices using " + wordWidth + " bits.");
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
	 */
	public ProbBackoffValueContainer(int valueRadix, boolean storePrefixIndexes, int maxNgramOrder, LongToIntHashMap hasBackoffValIndexer,
		LongToIntHashMap noBackoffValIndexer, long[] probsAndBackoffsForRank, float[] probsForRank, int wordWidth) {
		super(valueRadix, storePrefixIndexes, maxNgramOrder);
		this.hasBackoffValIndexer = hasBackoffValIndexer;
		this.noBackoffValIndexer = noBackoffValIndexer;
		this.probsAndBackoffsForRank = probsAndBackoffsForRank;
		this.probsForRank = probsForRank;
		super.wordWidth = wordWidth;
	}

	@Override
	public ProbBackoffValueContainer createFreshValues() {
		return new ProbBackoffValueContainer(valueRadix, storeSuffixIndexes, valueRanks.length, hasBackoffValIndexer, noBackoffValIndexer,
			probsAndBackoffsForRank, probsForRank,wordWidth);
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
		if (rank % 2 == HAS_BACKOFF)
			return backoff ? ProbBackoffPair.backoffOf(probsAndBackoffsForRank[rank >> 1]) : ProbBackoffPair.probOf(probsAndBackoffsForRank[rank >> 1]);
		else
			return backoff ? 0.0f : probsForRank[rank >> 1];
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

	@Override
	public void setFromOtherValues(final ValueContainer<ProbBackoffPair> o) {
		super.setFromOtherValues(o);
		this.hasBackoffValIndexer = ((ProbBackoffValueContainer) o).hasBackoffValIndexer;
		this.noBackoffValIndexer = ((ProbBackoffValueContainer) o).noBackoffValIndexer;
	}

	@Override
	public void trim() {
		super.trim();
		hasBackoffValIndexer = null;
		noBackoffValIndexer = null;
	}

	@Override
	protected int getCountRank(long val) {
		if (ProbBackoffPair.backoffOf(val) == 0.0f) {
			int rank = noBackoffValIndexer.get(val, -1);
			assert rank >= 0;
			return ((rank << 1) | NO_BACKOFF);
		} else {
			int rank = hasBackoffValIndexer.get(val, -1);
			if (rank < 0) {
				@SuppressWarnings("unused")
				int x = 5;
			}
			assert rank >= 0;

			return ((rank << 1) | HAS_BACKOFF);
		}
	}
}