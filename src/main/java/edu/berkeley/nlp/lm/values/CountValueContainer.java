package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap.Entry;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.LongRef;

public final class CountValueContainer extends RankedValueContainer<LongRef>
{

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	private final long[] countsForRank;

	private transient LongToIntHashMap countIndexer;

	private long unigramSum = 0L;

	public CountValueContainer(final LongToIntHashMap countCounter, final int valueRadix, final boolean storePrefixes, final long[] numNgramsForEachOrder) {
		super(valueRadix, storePrefixes, numNgramsForEachOrder);
		final boolean hasDefaultVal = countCounter.get(getDefaultVal().asLong(), -1) >= 0;
		countsForRank = new long[countCounter.size() + (hasDefaultVal ? 0 : 1)];
		countIndexer = new LongToIntHashMap();
		int k = 0;
		for (final Entry pair : countCounter.getObjectsSortedByValue(true)) {

			countIndexer.put(pair.key, countIndexer.size());
			countsForRank[k++] = pair.key;
			if (countIndexer.size() == defaultValRank && !hasDefaultVal) {
				countIndexer.put(getDefaultVal().asLong(), countIndexer.size());
				countsForRank[k++] = getDefaultVal().asLong();

			}
		}
		if (countIndexer.size() < defaultValRank && !hasDefaultVal) {
			countIndexer.put(getDefaultVal().asLong(), countIndexer.size());
			countsForRank[k++] = getDefaultVal().asLong();

		}
		valueWidth = CustomWidthArray.numBitsNeeded(countIndexer.size());
	}

	/**
	 * @param valueRadix
	 * @param storePrefixIndexes
	 * @param maxNgramOrder
	 * @param countsForRank
	 * @param countIndexer
	 */
	private CountValueContainer(int valueRadix, boolean storePrefixIndexes, long[] numNgramsForEachOrder, long[] countsForRank, LongToIntHashMap countIndexer,
		int wordWidth) {
		super(valueRadix, storePrefixIndexes, numNgramsForEachOrder);
		this.countsForRank = countsForRank;
		this.countIndexer = countIndexer;
		this.valueWidth = wordWidth;
	}

	@Override
	public CountValueContainer createFreshValues(long[] numNgramsForEachOrder_) {
		return new CountValueContainer(valueRadix, storeSuffixIndexes, numNgramsForEachOrder_, countsForRank, countIndexer, valueWidth);
	}

	@Override
	public void getFromOffset(final long index, final int ngramOrder, @OutputParameter final LongRef outputVal) {

		outputVal.value = getCount(ngramOrder, index, countsForRank);
	}

	@Override
	protected void getFromRank(final long rank, @OutputParameter final LongRef outputVal) {

		outputVal.value = countsForRank[(int) rank];
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
		final int countIndex = (int) valueRanks[ngramOrder].get(index);
		return array[countIndex];
	}

	@Override
	protected LongRef getDefaultVal() {
		return new LongRef(-1L);
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {
		super.trimAfterNgram(ngramOrder, size);
		if (ngramOrder == 0) {
			for (int i = 0; i < valueRanks[ngramOrder].size(); ++i) {
				unigramSum += countsForRank[(int) valueRanks[ngramOrder].get(i)];
			}
		}
	}

	public long getUnigramSum() {
		return unigramSum;
	}

	@Override
	public LongRef getScratchValue() {
		return new LongRef(-1);
	}

	@Override
	public void setFromOtherValues(final ValueContainer<LongRef> o) {
		super.setFromOtherValues(o);
		this.countIndexer = ((CountValueContainer) o).countIndexer;
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

}