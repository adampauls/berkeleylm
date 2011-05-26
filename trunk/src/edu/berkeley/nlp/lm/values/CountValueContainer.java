package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.LongRef;

public final class CountValueContainer extends LmValueContainer<LongRef>
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	private long[] countsForRank;

	private long unigramSum = 0L;

	public CountValueContainer(final Indexer<LongRef> countIndexer, final int valueRadix, final boolean storePrefixes) {
		super(countIndexer, valueRadix, storePrefixes);
	}

	@Override
	public CountValueContainer createFreshValues() {
		return new CountValueContainer(countIndexer, valueRadix, storePrefixIndexes);
	}

	@Override
	public void getFromOffset(final long index, final int ngramOrder, @OutputParameter final LongRef outputVal) {

		outputVal.value = getCount(ngramOrder, index, countsForRank);
	}

	@Override
	protected void getFromRank(final int rank, @OutputParameter final LongRef outputVal) {

		outputVal.value = countsForRank[rank];
	}

	public final long getCount(final int ngramOrder, final long index) {
		return getCount(ngramOrder, index, countsForRank);
	}

	/**
	 * @param ngramOrder
	 * @param index
	 * @param uncompressProbs2
	 * @return
	 */
	private long getCount(final int ngramOrder, final long index, final long[] array) {
		final int countIndex = (int) valueRanksCompressed[ngramOrder].get(index);
		return array[countIndex];
	}

	@Override
	protected LongRef getDefaultVal() {
		return new LongRef(-1L);
	}

	@Override
	protected void storeCounts() {
		countsForRank = new long[countIndexer.size()];
		int k = 0;
		for (final LongRef pair : countIndexer.getObjects()) {

			final int i = k;
			k++;

			countsForRank[i] = pair.value;
		}
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {
		super.trimAfterNgram(ngramOrder, size);
		if (ngramOrder == 0) {
			for (int i = 0; i < valueRanksCompressed[ngramOrder].size(); ++i) {
				unigramSum += countsForRank[(int) valueRanksCompressed[ngramOrder].get(i)];
			}
		}
	}

	public long getUnigramSum() {
		return unigramSum;
	}

}