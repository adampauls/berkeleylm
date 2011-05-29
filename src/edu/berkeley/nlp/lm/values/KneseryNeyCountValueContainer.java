package edu.berkeley.nlp.lm.values;

import java.util.Arrays;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.bits.BitList;
import edu.berkeley.nlp.lm.bits.BitStream;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

public final class KneseryNeyCountValueContainer implements ValueContainer<KneseryNeyCountValueContainer.KneserNeyCounts>
{

	public static class KneserNeyCounts
	{
		public long tokenCounts; // only stored for the highest- and second-highest-order n-grams

		public int leftDotTypeCounts; // N_{1+}(\cdot w) as in Chen and Goodman (1998), not stored for highest-order

		public int rightDotTypeCounts; // N_{1+}(w \cdot) as in Chen and Goodman (1998), not stored for highest-order

		public int dotdotTypeCounts; // N_{1+}(\dot w \dot) as in Chen and Goodman (1998), not stored for highest-order
	}

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	private LongArray tokenCounts;

	private LongArray prefixTokenCounts;

	@PrintMemoryCount
	private final LongArray[] rightDotTypeCounts;

	@PrintMemoryCount
	private final LongArray[] dotdotTypeCounts;

	@PrintMemoryCount
	private final LongArray[] leftDotTypeCounts;

	private long bigramTypeCounts = 0;

	private HashNgramMap<KneserNeyCounts> map;

	public KneseryNeyCountValueContainer(int maxNgramOrder) {
		this.tokenCounts = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, Integer.MAX_VALUE);
		this.prefixTokenCounts = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, Integer.MAX_VALUE);
		rightDotTypeCounts = new LongArray[maxNgramOrder - 1];
		leftDotTypeCounts = new LongArray[maxNgramOrder - 1];
		dotdotTypeCounts = new LongArray[maxNgramOrder - 1];
		for (int i = 0; i < maxNgramOrder - 1; ++i) {
			rightDotTypeCounts[i] = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, Integer.MAX_VALUE);
			leftDotTypeCounts[i] = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, Integer.MAX_VALUE);
			dotdotTypeCounts[i] = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, Integer.MAX_VALUE);
		}
	}

	@Override
	public KneseryNeyCountValueContainer createFreshValues() {
		final KneseryNeyCountValueContainer kneseryNeyCountValueContainer = new KneseryNeyCountValueContainer(rightDotTypeCounts.length + 1);

		return kneseryNeyCountValueContainer;
	}

	@Override
	public void getFromOffset(final long offset, final int ngramOrder, @OutputParameter final KneserNeyCounts outputVal) {
		final boolean isHighestOrder = isHighestOrder(ngramOrder);
		final boolean isSecondHighestOrder = isSecondHighestOrder(ngramOrder);
		outputVal.tokenCounts = isHighestOrder ? tokenCounts.get(offset) : (isSecondHighestOrder ? prefixTokenCounts.get(offset) : -1);
		outputVal.rightDotTypeCounts = (int) (isHighestOrder ? -1 : rightDotTypeCounts[ngramOrder].get(offset));
		outputVal.leftDotTypeCounts = (int) (isHighestOrder ? -1 : leftDotTypeCounts[ngramOrder].get(offset));
		outputVal.dotdotTypeCounts = (int) (isHighestOrder ? -1 : dotdotTypeCounts[ngramOrder].get(offset));
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {

	}

	@Override
	public KneserNeyCounts getScratchValue() {
		return new KneserNeyCounts();
	}

	@Override
	public void add(int[] ngram, int startPos, int endPos, int ngramOrder, long offset, long contextOffset, int word, KneserNeyCounts val, long suffixOffset,
		boolean ngramIsNew) {

		if (Arrays.toString(Arrays.copyOfRange(ngram, startPos, endPos)).contains("7") && ngramOrder == 1) {
			@SuppressWarnings("unused")
			int x = 5;
		}
		if (isHighestOrder(ngramOrder)) {
			if (val != null) {
				tokenCounts.set(offset, val.tokenCounts);
			} else {
				tokenCounts.incrementCount(offset, 1);
			}
		} else if (isSecondHighestOrder(ngramOrder)) {
			if (val != null) {
				prefixTokenCounts.set(offset, val.tokenCounts);
			} else {
				prefixTokenCounts.incrementCount(offset, 1);
			}
		}
		if (ngramIsNew) {
			if (ngramOrder > 0) {
				if (ngramOrder == 1) {
					bigramTypeCounts++;
				} else {
					LmContextInfo dotDotOffset = map.getOffsetForNgram(ngram, startPos + 1, endPos - 1);
					dotdotTypeCounts[ngramOrder - 2].incrementCount(dotDotOffset.offset, 1);
				}
				LmContextInfo leftDotOffset = map.getOffsetForNgram(ngram, startPos + 1, endPos);
				leftDotTypeCounts[ngramOrder - 1].incrementCount(leftDotOffset.offset, 1);
				LmContextInfo rightDotOffset = map.getOffsetForNgram(ngram, startPos, endPos - 1);
				rightDotTypeCounts[ngramOrder - 1].incrementCount(rightDotOffset.offset, 1);
			}
		}
	}

	@Override
	public void swap(long a, long b, int ngramOrder) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

	@Override
	public void setSizeAtLeast(long size, int ngramOrder) {
		if (isHighestOrder(ngramOrder)) {
			tokenCounts.setAndGrowIfNeeded(size - 1, 0);
		} else {
			if (isSecondHighestOrder(ngramOrder)) prefixTokenCounts.setAndGrowIfNeeded(size - 1, 0);
			leftDotTypeCounts[ngramOrder].setAndGrowIfNeeded(size - 1, 0);
			rightDotTypeCounts[ngramOrder].setAndGrowIfNeeded(size - 1, 0);
			dotdotTypeCounts[ngramOrder].setAndGrowIfNeeded(size - 1, 0);

		}

	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private boolean isHighestOrder(int ngramOrder) {
		return ngramOrder == dotdotTypeCounts.length;
	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private boolean isSecondHighestOrder(int ngramOrder) {
		return ngramOrder == dotdotTypeCounts.length - 1;
	}

	@Override
	public void setFromOtherValues(ValueContainer<KneserNeyCounts> other) {
		final KneseryNeyCountValueContainer other_ = (KneseryNeyCountValueContainer) other;
		tokenCounts = other_.tokenCounts;
		System.arraycopy(other_.dotdotTypeCounts, 0, dotdotTypeCounts, 0, dotdotTypeCounts.length);
		System.arraycopy(other_.rightDotTypeCounts, 0, rightDotTypeCounts, 0, rightDotTypeCounts.length);
		System.arraycopy(other_.leftDotTypeCounts, 0, leftDotTypeCounts, 0, leftDotTypeCounts.length);
		prefixTokenCounts = other_.prefixTokenCounts;
		bigramTypeCounts = other_.bigramTypeCounts;
	}

	@Override
	public BitList getCompressed(long offset, int ngramOrder) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

	@Override
	public void decompress(BitStream bits, int ngramOrder, boolean justConsume, KneserNeyCounts outputVal) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

	@Override
	public void clearStorageAfterCompression(int ngramOrder) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

	@Override
	public void trim() {
		tokenCounts.trim();
		prefixTokenCounts.trim();

		for (int i = 0; i < rightDotTypeCounts.length; ++i) {
			rightDotTypeCounts[i].trim();
			leftDotTypeCounts[i].trim();
			dotdotTypeCounts[i].trim();
		}
	}

	@Override
	public long getContextOffset(long offset, int ngramOrder) {
		throw new UnsupportedOperationException("Method not yet implemented");
	}

	@Override
	public void initForMap(NgramMap<KneserNeyCounts> map) {
		this.map = (HashNgramMap<KneserNeyCounts>) map;
	}

}