package edu.berkeley.nlp.lm.values;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

/**
 * Stored type and token counts necessary for estimating a Kneser-Ney language
 * model
 * 
 * @author adampauls
 * 
 */
public final class KneserNeyCountValueContainer implements ValueContainer<KneserNeyCountValueContainer.KneserNeyCounts>
{

	/**
	 * Warning: type counts are stored internally as 32-bit ints.
	 * 
	 * @author adampauls
	 * 
	 */
	public static class KneserNeyCounts
	{
		public long tokenCounts; // only stored for the highest- and second-highest-order n-grams

		public long leftDotTypeCounts; // N_{1+}(\cdot w) as in Chen and Goodman (1998), not stored for highest-order

		public long rightDotTypeCounts; // N_{1+}(w \cdot) as in Chen and Goodman (1998), not stored for highest-order

		public long dotdotTypeCounts; // N_{1+}(\dot w \dot) as in Chen and Goodman (1998), not stored for highest-order
	}

	private static final long serialVersionUID = 964277160049236607L;

	@PrintMemoryCount
	private LongArray tokenCounts; // for highest-order ngrams

	private LongArray prefixTokenCounts;// for second-highest order n-grams

	@PrintMemoryCount
	private final LongArray[] rightDotTypeCounts;

	@PrintMemoryCount
	private final LongArray[] dotdotTypeCounts;

	@PrintMemoryCount
	private final LongArray[] leftDotTypeCounts;

	private long bigramTypeCounts = 0;

	private HashNgramMap<KneserNeyCounts> map;

	public KneserNeyCountValueContainer(final int maxNgramOrder) {
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
	public KneserNeyCountValueContainer createFreshValues() {
		final KneserNeyCountValueContainer kneseryNeyCountValueContainer = new KneserNeyCountValueContainer(rightDotTypeCounts.length + 1);

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
	public boolean add(final int[] ngram, final int startPos, final int endPos, final int ngramOrder, final long offset, final long contextOffset,
		final int word, final KneserNeyCounts val, final long suffixOffset, final boolean ngramIsNew) {

		assert !map.isReversed();
		if (isHighestOrder(ngramOrder)) {
			tokenCounts.incrementCount(offset, val.tokenCounts);
		} else if (isSecondHighestOrder(ngramOrder)) {
			prefixTokenCounts.incrementCount(offset, val.tokenCounts);
		}
		if (ngramIsNew) {
			if (ngramOrder > 0) {
				if (ngramOrder == 1) {
					bigramTypeCounts++;
				} else {
					final long dotDotOffset = map.getPrefixOffset(suffixOffset, endPos - startPos - 2);//map.getOffsetForNgramInModel(ngram, startPos + 1, endPos - 1);
					dotdotTypeCounts[ngramOrder - 2].incrementCount(dotDotOffset, 1);
				}
				final long leftDotOffset = suffixOffset; //map.getOffsetForNgramInModel(ngram, startPos + 1, endPos);
				leftDotTypeCounts[ngramOrder - 1].incrementCount(leftDotOffset, 1);
				final long rightDotOffset = contextOffset;//map.getOffsetForNgramInModel(ngram, startPos, endPos - 1);
				rightDotTypeCounts[ngramOrder - 1].incrementCount(rightDotOffset, 1);
			}
		}
		return true;
	}

	@Override
	public void setSizeAtLeast(final long size, final int ngramOrder) {
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
	private boolean isHighestOrder(final int ngramOrder) {
		return ngramOrder == dotdotTypeCounts.length;
	}

	/**
	 * @param ngramOrder
	 * @return
	 */
	private boolean isSecondHighestOrder(final int ngramOrder) {
		return ngramOrder == dotdotTypeCounts.length - 1;
	}

	@Override
	public void setFromOtherValues(final ValueContainer<KneserNeyCounts> other) {
		final KneserNeyCountValueContainer other_ = (KneserNeyCountValueContainer) other;
		tokenCounts = other_.tokenCounts;
		System.arraycopy(other_.dotdotTypeCounts, 0, dotdotTypeCounts, 0, dotdotTypeCounts.length);
		System.arraycopy(other_.rightDotTypeCounts, 0, rightDotTypeCounts, 0, rightDotTypeCounts.length);
		System.arraycopy(other_.leftDotTypeCounts, 0, leftDotTypeCounts, 0, leftDotTypeCounts.length);
		prefixTokenCounts = other_.prefixTokenCounts;
		bigramTypeCounts = other_.bigramTypeCounts;
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
	public void setMap(final NgramMap<KneserNeyCounts> map) {
		this.map = (HashNgramMap<KneserNeyCounts>) map;
	}

	@Override
	public void clearStorageForOrder(int ngramOrder) {
		if (ngramOrder == rightDotTypeCounts.length) {
			tokenCounts = null;
		} else if (ngramOrder == rightDotTypeCounts.length - 1) {
			prefixTokenCounts = null;
		}
		if (ngramOrder < rightDotTypeCounts.length) {
			rightDotTypeCounts[ngramOrder] = null;
			leftDotTypeCounts[ngramOrder] = null;
			dotdotTypeCounts[ngramOrder] = null;
		}
	}

}