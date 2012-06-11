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

	//	private transient LongToIntHashMap hasBackoffValIndexer;

	@PrintMemoryCount
	float[] backoffsForRank;

	//	@PrintMemoryCount
	//	final CustomWidthArray probsAndBackoffsForRank; // ugly, but we but probs and backoffs consecutively in this area to improve cache locality

	@PrintMemoryCount
	final float[] probsForRank; // ugly, but we but probs and backoffs consecutively in this area to improve cache locality

	private int backoffWidth = -1;

	Indexer<Float> probIndexer = new Indexer<Float>();

	Indexer<Float> backoffIndexer = new Indexer<Float>();

	public CompressibleProbBackoffValueContainer(final LongToIntHashMap countCounter, final int valueRadix, final boolean storePrefixes,
		long[] numNgramsForEachOrder) {
		super(valueRadix, storePrefixes, numNgramsForEachOrder);
		Logger.startTrack("Storing values");
		//		final boolean hasDefaultVal = countCounter.get(getDefaultVal().asLong(), -1) >= 0;
		//		hasBackoffValIndexer = new LongToIntHashMap();
		//		noBackoffValIndexer = new LongToIntHashMap();
		List<Entry> objectsSortedByValue = countCounter.getObjectsSortedByValue(true);

		LongToIntHashMap probSorter = new LongToIntHashMap();
		LongToIntHashMap backoffSorter = new LongToIntHashMap();
		for (Entry e : objectsSortedByValue) {
			probSorter.incrementCount(Float.floatToIntBits(ProbBackoffPair.probOf(e.key)) & ((1L << Integer.SIZE) - 1), e.value);
			backoffSorter.incrementCount(Float.floatToIntBits(ProbBackoffPair.backoffOf(e.key)) & ((1L << Integer.SIZE) - 1), e.value);
		}
		for (Entry probEntry : probSorter.getObjectsSortedByValue(true)) {
			probIndexer.getIndex(Float.intBitsToFloat((int) probEntry.key));
			if (probIndexer.size() == defaultValRank) {
				probIndexer.getIndex(getDefaultVal().prob);
			}
		}
		if (probIndexer.size() < defaultValRank) {
			probIndexer.getIndex(getDefaultVal().prob);
		}
		for (Entry backoffEntry : backoffSorter.getObjectsSortedByValue(true)) {
			backoffIndexer.getIndex(Float.intBitsToFloat((int) backoffEntry.key));
			if (backoffIndexer.size() == defaultValRank) {
				backoffIndexer.getIndex(getDefaultVal().backoff);
			}
		}
		if (backoffIndexer.size() < defaultValRank) {
			backoffIndexer.getIndex(getDefaultVal().backoff);
		}
		//		//		hasBackoffValIndexer = new LongToIntHashMap();
		//		for (Entry e : objectsSortedByValue) {
		//			final float backoff = ProbBackoffPair.backoffOf(e.key);
		//			final float prob = ProbBackoffPair.probOf(e.key);
		//			probIndexer.getIndex(prob);
		//			backoffIndexer.getIndex(backoff);
		//		}
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
		wordWidth = CustomWidthArray.numBitsNeeded(probIndexer.size()) + backoffWidth;
		//		final int width = 
		////		probsAndBackoffsForRank = new CustomWidthArray(objectsSortedByValue.size() + (hasDefaultVal ? 0 : 1), width);
		////		probsAndBackoffsForRank.ensureCapacity(objectsSortedByValue.size() + (hasDefaultVal ? 0 : 1));
		//		for (Entry e : objectsSortedByValue) {
		//
		//			final float backoff = ProbBackoffPair.backoffOf(e.key);
		//			final float prob = ProbBackoffPair.probOf(e.key);
		//			int probIndex = probIndexer.getIndex(prob);
		//			int backoffIndex = backoffIndexer.getIndex(backoff);
		//			long together = combine(probIndex, backoffIndex);
		//			hasBackoffValIndexer.put(e.key, (int) probsAndBackoffsForRank.size());
		//			probsAndBackoffsForRank.addWithFixedCapacity(together);
		//
		//			if (probsAndBackoffsForRank.size() == defaultValRank && !hasDefaultVal) {
		//				addDefault(probIndexer, backoffIndexer);
		//
		//			}
		//			//			if (backoff == 0.0f) {
		//			//				noBackoffValIndexer.put(e.key, noBackoffValIndexer.size());
		//			//			} else {
		//			//				hasBackoffValIndexer.put(e.key, hasBackoffValIndexer.size());
		//			//			}
		//		}
		//		if (probsAndBackoffsForRank.size() < defaultValRank && !hasDefaultVal) {
		//			addDefault(probIndexer, backoffIndexer);
		//
		//		}
		//
		//		wordWidth = CustomWidthArray.numBitsNeeded(probsAndBackoffsForRank.size());
		//		for (java.util.Map.Entry<Long, Integer> entry : hasBackoffValIndexer.entries()) {
		//			probsAndBackoffsForRank[entry.getValue()] = entry.getKey();
		//		}
		//		for (java.util.Map.Entry<Long, Integer> entry : noBackoffValIndexer.entries()) {
		//			probsForRank[(entry.getValue())] = ProbBackoffPair.probOf(entry.getKey());
		//		}

		Logger.logss("Storing count indices using " + wordWidth + " bits.");
		Logger.endTrack();
	}

	/**
	 * @param probIndexer
	 * @param backoffIndexer
	 * @param backoffWidth
	 * @param k
	 * @return
	 */
	//	private void addDefault(Indexer<Float> probIndexer, Indexer<Float> backoffIndexer) {
	//		final float dbackoff = ProbBackoffPair.backoffOf(getDefaultVal().asLong());
	//		final float dprob = ProbBackoffPair.probOf(getDefaultVal().asLong());
	//		int dprobIndex = probIndexer.getIndex(dprob);
	//		int dbackoffIndex = backoffIndexer.getIndex(dbackoff);
	//		long dtogether = combine(dprobIndex, dbackoffIndex);
	//		hasBackoffValIndexer.put(getDefaultVal().asLong(), (int) probsAndBackoffsForRank.size());
	//		probsAndBackoffsForRank.addWithFixedCapacity(dtogether);
	//	}

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
		//		this.probsAndBackoffsForRank = probsAndBackoffsForRank;
		this.probsForRank = probsForRank;
		super.wordWidth = wordWidth;
		//		this.hasBackoffValIndexer = hasBackoffValIndexer;
		this.backoffWidth = backoffWidth;
	}

	@Override
	public CompressibleProbBackoffValueContainer createFreshValues(long[] numNgramsForEachOrder_) {
		return new CompressibleProbBackoffValueContainer(valueRadix, storeSuffixIndexes, numNgramsForEachOrder_, probsForRank, backoffsForRank, probIndexer,
			wordWidth, backoffIndexer, backoffWidth);
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
		final long rank = getRank(valueRanksForOrder, index);
		return getFromRank(rank, backoff);
	}

	private float getFromRank(final long val, final boolean backoff) {
		//		long val = probsAndBackoffsForRank.get(rank);
		return backoff ? backoffsForRank[backoffRankOf(val)] : probsForRank[probRankOf(val)];

		//		if (rank % 2 == HAS_BACKOFF)
		//			return backoff ? ProbBackoffPair.backoffOf(probsAndBackoffsForRank[rank >> 1]) : ProbBackoffPair.probOf(probsAndBackoffsForRank[rank >> 1]);
		//		else
		//			return backoff ? 0.0f : probsForRank[rank >> 1];
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
		//		long val = probsAndBackoffsForRank.get(rank);
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