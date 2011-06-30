package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

public final class ProbBackoffValueContainer extends LmValueContainer<ProbBackoffPair>
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	float[] probsForRank;


	public ProbBackoffValueContainer(final Indexer<ProbBackoffPair> countIndexer, final int valueRadix, final boolean storePrefixes) {
		super(countIndexer, valueRadix, storePrefixes);
	}

	@Override
	public ProbBackoffValueContainer createFreshValues() {
		return new ProbBackoffValueContainer(countIndexer, valueRadix, storePrefixIndexes);
	}

	public final float getProb(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, 0);
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
	private float getCount(final int ngramOrder, final long index, int shiftBits) {
		final int rank = getRank(ngramOrder, index);
		return getFromRank(rank, shiftBits);
	}

	private float getFromRank(int rank, int shiftBits) {
		return probsForRank[2*rank+(shiftBits > 0 ? 1 : 0)];
		//return Float.intBitsToFloat((int)(probsForRank[rank] >>> shiftBits));
	}

	public final float getBackoff(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, Integer.SIZE);
	}

	@Override
	protected ProbBackoffPair getDefaultVal() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}

	@Override
	protected void storeCounts() {
		probsForRank = new float[2*countIndexer.size()];
		int k = 0;
		for (final ProbBackoffPair pair : countIndexer.getObjects()) {

			probsForRank[k] = pair.prob;
			probsForRank[k+1] = pair.backoff;
			// | (((long)Float.floatToIntBits(pair.backoff)) << Integer.SIZE);
			k++;
			k++;
		}
	}

	@Override
	protected void getFromRank(final int rank, @OutputParameter final ProbBackoffPair outputVal) {

		outputVal.prob = getFromRank(rank, 0);//probsForRank[rank];
		outputVal.backoff = getFromRank(rank, Integer.SIZE);
	}

	@Override
	public ProbBackoffPair getScratchValue() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}

}
