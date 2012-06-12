package edu.berkeley.nlp.lm.phrasetable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;
import edu.berkeley.nlp.lm.values.ValueContainer;

/**
 * Stored type and token counts necessary for estimating a Kneser-Ney language
 * model
 * 
 * @author adampauls
 * 
 */
public final class PhraseTableValueContainer implements ValueContainer<PhraseTableValueContainer.PhraseTableValues>
{
	private static final long serialVersionUID = 964277160049236607L;

	private static final int EMPTY_VALUE_INDEX = Integer.MAX_VALUE;

	public interface PhraseTableValues extends Serializable
	{

	}

	public static class FeaturePhraseTableValues implements PhraseTableValues
	{
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		float[] features;

		public FeaturePhraseTableValues(final float[] features) {
			this.features = features;
		}

	}

	public static class TargetTranslationsValues implements PhraseTableValues
	{
		/**
		 * 
		 */
		private static final long serialVersionUID = 1L;

		long[] targetTranslationOffsets;

		int[] targetTranslationOrders;
	}

	@PrintMemoryCount
	private LongArray[] features;

	@PrintMemoryCount
	private LongArray[] valueIndexes;

	@PrintMemoryCount
	private ArrayList<CustomWidthArray>[] targetTranslations;

	private HashNgramMap<PhraseTableValues> map;

	private final int separatorWord;

	private final int numFeatures;

	@SuppressWarnings("unchecked")
	public PhraseTableValueContainer(final int separatorWord, final int numFeatures) {
		this.separatorWord = separatorWord;
		this.numFeatures = numFeatures;
		this.targetTranslations = new ArrayList[5];
		this.valueIndexes = new LongArray[5];
		this.features = new LongArray[5];
	}

	@Override
	public PhraseTableValueContainer createFreshValues(long[] numNgramsForEachOrder_) {
		return new PhraseTableValueContainer(separatorWord, numFeatures);
	}

	@Override
	public void getFromOffset(final long offset, final int ngramOrder, @OutputParameter final PhraseTableValues outputVal) {
		if (offset >= valueIndexes[ngramOrder].size()) return;
		final long valueIndex = valueIndexes[ngramOrder].get(offset);
		if (valueIndex == EMPTY_VALUE_INDEX) return;
		if (outputVal instanceof FeaturePhraseTableValues && valueIndex >= 0) {
			final float[] fs = new float[numFeatures];
			for (int i = 0; i < numFeatures; ++i)
				fs[i] = Float.intBitsToFloat((int) features[ngramOrder].get((int) (valueIndex + i)));
			((FeaturePhraseTableValues) outputVal).features = fs;
		}
		if (outputVal instanceof TargetTranslationsValues && valueIndex < 0) {
			((TargetTranslationsValues) outputVal).targetTranslationOffsets = readOffsets(targetTranslations[ngramOrder].get((int) (-valueIndex - 1)));
			((TargetTranslationsValues) outputVal).targetTranslationOrders = readOrders(targetTranslations[ngramOrder].get((int) (-valueIndex - 1)));
		}
	}

	private int[] readOrders(final CustomWidthArray longArray) {
		final int[] ret = new int[(int) longArray.size()];
		for (int i = 0; i < longArray.size(); ++i)
			ret[i] = (byte) (longArray.get(i) >> Integer.SIZE);
		return ret;
	}

	private long[] readOffsets(final CustomWidthArray longArray) {
		final long[] ret = new long[(int) longArray.size()];
		for (int i = 0; i < longArray.size(); ++i)
			ret[i] = (int) longArray.get(i);
		return ret;
	}

	@Override
	public void trimAfterNgram(final int ngramOrder, final long size) {

	}

	@Override
	public PhraseTableValues getScratchValue() {
		return new FeaturePhraseTableValues(null);
	}

	@Override
	public boolean add(final int[] ngram, final int startPos, final int endPos, final int ngramOrder, final long offset, final long contextOffset,
		final int word, final PhraseTableValues val, final long suffixOffset, final boolean ngramIsNew) {

		assert !map.isReversed();

		final boolean isSourceSidePhrase = !containsSeparator(ngram, startPos, endPos);
		if (isSourceSidePhrase) {
			addNewSrcPhrase(ngramOrder, offset);
		} else if (val instanceof FeaturePhraseTableValues && ((FeaturePhraseTableValues) val).features != null) {
			addFeaturesForWholePhrase(ngramOrder, offset, val);

			addPointerToTargetSidePhrase(ngramOrder, offset, contextOffset, word);
		} else if (ngramIsNew) {
			assert val instanceof TargetTranslationsValues || ((FeaturePhraseTableValues) val).features == null;
			growValueIndexArrayIfNecessary(ngramOrder);

			valueIndexes[ngramOrder].setAndGrowIfNeeded((int) (offset), EMPTY_VALUE_INDEX);
		}
		return true;

	}

	private boolean containsSeparator(final int[] ngram, final int startPos, final int endPos) {
		for (int i = startPos; i < endPos; ++i)
			if (ngram[i] == separatorWord) return true;
		return false;
	}

	/**
	 * @param ngramOrder
	 * @param offset
	 */
	private void addNewSrcPhrase(final int ngramOrder, final long offset) {
		growValueIndexArrayIfNecessary(ngramOrder);
		if (ngramOrder >= targetTranslations.length) {
			targetTranslations = Arrays.copyOf(targetTranslations, targetTranslations.length * 3 / 2);
		}
		if (targetTranslations[ngramOrder] == null) {
			targetTranslations[ngramOrder] = new ArrayList<CustomWidthArray>();
		}
		final ArrayList<CustomWidthArray> targetTranslationPointersHere = targetTranslations[ngramOrder];
		final long currVal = offset >= valueIndexes[ngramOrder].size() ? 0 : valueIndexes[ngramOrder].get((int) (offset));
		if (currVal == 0) valueIndexes[ngramOrder].setAndGrowIfNeeded((int) (offset), (-targetTranslations[ngramOrder].size() - 1));

		targetTranslationPointersHere.add(new CustomWidthArray(3, Integer.SIZE + Byte.SIZE));
	}

	/**
	 * @param ngramOrder
	 * @param offset
	 * @param contextOffset
	 * @param word
	 */
	private void addPointerToTargetSidePhrase(final int ngramOrder, final long offset, final long contextOffset, final int word) {
		int currWord = word;
		long srcPhraseOffset = contextOffset;
		int srcPhraseOrder = ngramOrder - 1;
		while (currWord != separatorWord) {
			currWord = map.getNextWord(srcPhraseOffset, srcPhraseOrder);
			srcPhraseOffset = map.getNextContextOffset(srcPhraseOffset, srcPhraseOrder);
			srcPhraseOrder--;
		}

		final long valueIndex = -valueIndexes[srcPhraseOrder].get(srcPhraseOffset) - 1;
		final ArrayList<CustomWidthArray> targetTranslationPointersHere = targetTranslations[srcPhraseOrder];
		targetTranslationPointersHere.get((int) valueIndex).add(combineOrderAndOffset(ngramOrder, offset));
	}

	/**
	 * @param ngramOrder
	 * @param offset
	 * @return
	 */
	private long combineOrderAndOffset(final int ngramOrder, final long offset) {
		return (((long) ngramOrder) << Integer.SIZE) | offset;
	}

	/**
	 * @param ngramOrder
	 * @param offset
	 * @param val
	 */
	private void addFeaturesForWholePhrase(final int ngramOrder, final long offset, final PhraseTableValues val) {
		growValueIndexArrayIfNecessary(ngramOrder);
		if (ngramOrder >= features.length) {
			features = Arrays.copyOf(features, Math.max(ngramOrder + 1, features.length * 3 / 2));
		}
		if (features[ngramOrder] == null) features[ngramOrder] = LongArray.StaticMethods.newLongArray(Integer.MAX_VALUE, Integer.MAX_VALUE);
		valueIndexes[ngramOrder].setAndGrowIfNeeded((int) (offset), features[ngramOrder].size());
		for (int f = 0; f < numFeatures; ++f)
			features[ngramOrder].add(Float.floatToIntBits(((FeaturePhraseTableValues) val).features[f]));
	}

	/**
	 * @param ngramOrder
	 */
	private void growValueIndexArrayIfNecessary(final int ngramOrder) {
		if (ngramOrder >= valueIndexes.length) {
			valueIndexes = Arrays.copyOf(valueIndexes, Math.max(ngramOrder + 1, valueIndexes.length * 3 / 2));
		}
		if (valueIndexes[ngramOrder] == null) valueIndexes[ngramOrder] = LongArray.StaticMethods.newLongArray(Integer.MAX_VALUE, Integer.MAX_VALUE);
	}

	@Override
	public void setSizeAtLeast(final long size, final int ngramOrder) {
	}

	@Override
	public void setFromOtherValues(final ValueContainer<PhraseTableValues> other) {
		final PhraseTableValueContainer other_ = (PhraseTableValueContainer) other;
		this.features = other_.features;
		this.targetTranslations = other_.targetTranslations;
		this.valueIndexes = other_.valueIndexes;
	}

	@Override
	public void trim() {
		for (int ngramOrder = 0; ngramOrder < features.length; ++ngramOrder) {
			if (features[ngramOrder] != null) features[ngramOrder].trim();
			if (valueIndexes[ngramOrder] != null) valueIndexes[ngramOrder].trim();
			if (ngramOrder < targetTranslations.length && targetTranslations[ngramOrder] != null) {
				targetTranslations[ngramOrder].trimToSize();

				for (int j = 0; j < targetTranslations[ngramOrder].size(); ++j) {
					targetTranslations[ngramOrder].get(j).trim();
				}
			}
		}
	}

	@Override
	public void setMap(final NgramMap<PhraseTableValues> map) {
		this.map = (HashNgramMap<PhraseTableValues>) map;
	}

	public int getSeparatorWord() {
		return separatorWord;
	}

	@Override
	public void clearStorageForOrder(int ngramOrder) {
		features[ngramOrder] = null;
		valueIndexes[ngramOrder] = null;
		targetTranslations[ngramOrder] = null;
	}

	@Override
	public boolean storeSuffixoffsets() {
		return false;
	}

	@Override
	public int numValueBits(int ngramOrder) {
		return 0;
	}

}