package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

public final class ProbBackoffValueContainer extends LmValueContainer<ProbBackoffPair>
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	float[] probsForRank;

	@PrintMemoryCount
	float[] backoffsForRank;

	public ProbBackoffValueContainer(final Indexer<ProbBackoffPair> countIndexer, final int valueRadix, final boolean storePrefixes) {
		super(countIndexer, valueRadix, storePrefixes);
	}

	@Override
	public ProbBackoffValueContainer createFreshValues() {
		return new ProbBackoffValueContainer(countIndexer, valueRadix, storePrefixIndexes);
	}

	public final float getProb(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, probsForRank);
	}

	@Override
	public void getFromOffset(final long index, final int ngramOrder, @OutputParameter ProbBackoffPair outputVal) {
		final int rank = (int) valueRanksCompressed[ngramOrder].get(index);
		outputVal.prob = probsForRank[rank];
		outputVal.backoff = backoffsForRank[rank];
	}

	/**
	 * @param ngramOrder
	 * @param index
	 * @param uncompressProbs2
	 * @return
	 */
	private float getCount(final int ngramOrder, final long index, final float[] array) {
		final int rank = (int) valueRanksCompressed[ngramOrder].get(index);
		return array[rank];
	}

	public final float getBackoff(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, backoffsForRank);
	}

	@Override
	protected ProbBackoffPair getDefaultVal() {
		return new ProbBackoffPair(Float.NaN, Float.NaN);
	}

	@Override
	protected void storeCounts() {
		probsForRank = new float[countIndexer.size()];
		backoffsForRank = new float[countIndexer.size()];
		int k = 0;
		for (final ProbBackoffPair pair : countIndexer.getObjects()) {

			final int i = k;
			k++;

			probsForRank[i] = pair.prob;
			backoffsForRank[i] = pair.backoff;
		}
	}

	@Override
	protected void getFromRank(final int rank, @OutputParameter ProbBackoffPair outputVal) {

		outputVal.prob = probsForRank[rank];
		outputVal.backoff = backoffsForRank[rank];
	}

}