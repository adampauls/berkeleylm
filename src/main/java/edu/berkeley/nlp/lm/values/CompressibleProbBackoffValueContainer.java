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

public final class CompressibleProbBackoffValueContainer extends RankedValueContainer<ProbBackoffPair> implements ProbBackoffValueContainer
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	final float[] backoffsForRank;

	@PrintMemoryCount
	final float[] probsForRank; 

	private int backoffWidth = -1;

	private transient Indexer<Float> probIndexer = new Indexer<Float>();

	private transient Indexer<Float> backoffIndexer = new Indexer<Float>();

	public CompressibleProbBackoffValueContainer(final LongToIntHashMap countCounter, final int valueRadix, final boolean storePrefixes,
		long[] numNgramsForEachOrder) {
		super(valueRadix, storePrefixes, numNgramsForEachOrder);
		Logger.startTrack("Storing values");
		final boolean hasDefaultVal = countCounter.get(getDefaultVal().asLong(), -1) >= 0;
		List<Entry> objectsSortedByValue = countCounter.getObjectsSortedByValue(true);

		LongToIntHashMap probSorter = new LongToIntHashMap();
		LongToIntHashMap backoffSorter = new LongToIntHashMap();
		for (Entry e : objectsSortedByValue) {
			probSorter.incrementCount(Float.floatToIntBits(ProbBackoffPair.probOf(e.key)) & ((1L << Integer.SIZE) - 1), e.value);
			backoffSorter.incrementCount(Float.floatToIntBits(ProbBackoffPair.backoffOf(e.key)) & ((1L << Integer.SIZE) - 1), e.value);
		}
		for (Entry probEntry : probSorter.getObjectsSortedByValue(true)) {
			probIndexer.getIndex(Float.intBitsToFloat((int) probEntry.key));
			if (!hasDefaultVal && probIndexer.size() == defaultValRank) {
				probIndexer.getIndex(getDefaultVal().prob);
			}
		}
		if (!hasDefaultVal && probIndexer.size() < defaultValRank) {
			probIndexer.getIndex(getDefaultVal().prob);
		}
		for (Entry backoffEntry : backoffSorter.getObjectsSortedByValue(true)) {
			backoffIndexer.getIndex(Float.intBitsToFloat((int) backoffEntry.key));
			if (!hasDefaultVal && backoffIndexer.size() == defaultValRank) {
				backoffIndexer.getIndex(getDefaultVal().backoff);
			}
		}
		if (!hasDefaultVal && backoffIndexer.size() < defaultValRank) {
			backoffIndexer.getIndex(getDefaultVal().backoff);
		}
		
		probsForRank = new float[probIndexer.size()];
		int a = 0;
		for (float f : probIndexer.getObjects()) {
			probsForRank[a++] = f;
		}
		backoffsForRank = new float[backoffIndexer.size()];
		int b = 0;
		for (float f : backoffIndexer.getObjects()) {
			backoffsForRank[b++] = f;
		}
		backoffWidth = CustomWidthArray.numBitsNeeded(backoffIndexer.size());
		valueWidth = CustomWidthArray.numBitsNeeded(probIndexer.size()) + backoffWidth;
		

		Logger.logss("Storing count indices using " + valueWidth + " bits.");
		Logger.endTrack();
	}
	
	/**
	 * @param dprobIndex
	 * @param dbackoffIndex
	 * @return
	 */
	private long combine(int dprobIndex, int dbackoffIndex) {
		assert dprobIndex >= 0;
		assert dbackoffIndex >= 0;
		return (((long) dprobIndex) << backoffWidth) | dbackoffIndex;
	}

	private int backoffRankOf(long val) {
		return (int) (val & ((1L << backoffWidth) - 1));
	}

	private int probRankOf(long val) {
		return (int) (val >>> backoffWidth);
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
	public CompressibleProbBackoffValueContainer(int valueRadix, boolean storePrefixIndexes, long[] numNgramsForEachOrder, float[] probsForRank,
		float[] backoffsForRank, Indexer<Float> probIndexer, int wordWidth, Indexer<Float> backoffIndexer, int backoffWidth) {
		super(valueRadix, storePrefixIndexes, numNgramsForEachOrder);
		this.backoffsForRank = backoffsForRank;
		this.probIndexer = probIndexer;
		this.backoffIndexer = backoffIndexer;
		this.probsForRank = probsForRank;
		super.valueWidth = wordWidth;
		this.backoffWidth = backoffWidth;
	}

	@Override
	public CompressibleProbBackoffValueContainer createFreshValues(long[] numNgramsForEachOrder_) {
		return new CompressibleProbBackoffValueContainer(valueRadix, storeSuffixIndexes, numNgramsForEachOrder_, probsForRank, backoffsForRank, probIndexer,
			valueWidth, backoffIndexer, backoffWidth);
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

	private float getFromRank(final long val, final boolean backoff) {
		return backoff ? backoffsForRank[backoffRankOf(val)] : probsForRank[probRankOf(val)];
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
	protected void getFromRank(final long rank, @OutputParameter final ProbBackoffPair outputVal) {

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
		this.backoffIndexer = ((CompressibleProbBackoffValueContainer) o).backoffIndexer;
		this.probIndexer = ((CompressibleProbBackoffValueContainer) o).probIndexer;
	}

	@Override
	public void trim() {
		super.trim();
		backoffIndexer = probIndexer = null;
	}

	@Override
	protected long getCountRank(long val) {
		return combine(probIndexer.getIndex(ProbBackoffPair.probOf(val)), backoffIndexer.getIndex(ProbBackoffPair.backoffOf(val)));
	}

	@Override
	public BitList getCompressed(final long offset, final int ngramOrder) {
		final long rank = getRank(ngramOrder, offset);
		final BitList probBits = valueCoder.compress(probRankOf(rank));
		if (ngramOrder < numNgramsForEachOrder.length - 1) probBits.addAll(valueCoder.compress(backoffRankOf(rank)));
		return probBits;
	}

	@Override
	public final void decompress(final BitStream bits, final int ngramOrder, final boolean justConsume, @OutputParameter final ProbBackoffPair outputVal) {
		final long probRank = valueCoder.decompress(bits);
		final long backoffRank = (ngramOrder < numNgramsForEachOrder.length - 1) ? valueCoder.decompress(bits) : -1;
		if (justConsume) return;
		if (outputVal != null) {
			outputVal.prob = probsForRank[(int) probRank];
			outputVal.backoff = (ngramOrder < numNgramsForEachOrder.length - 1) ? backoffsForRank[(int) backoffRank] : 0;
		}
	}

}