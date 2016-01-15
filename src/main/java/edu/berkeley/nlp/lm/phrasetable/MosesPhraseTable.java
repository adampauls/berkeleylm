package edu.berkeley.nlp.lm.phrasetable;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.phrasetable.PhraseTableValueContainer.FeaturePhraseTableValues;
import edu.berkeley.nlp.lm.phrasetable.PhraseTableValueContainer.PhraseTableValues;
import edu.berkeley.nlp.lm.phrasetable.PhraseTableValueContainer.TargetTranslationsValues;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * 
 * Experimental class for reading Moses phrase tables and storing them
 * efficiently in memory using a trie.
 * 
 * @author adampauls
 * 
 */
public class MosesPhraseTable implements Serializable
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public static class TargetSideTranslation
	{

		// only stores the first 4 features from a moses file (i.e. does not store the bias)
		public float[] features;

		public int[] trgWords;

		@Override
		public String toString() {
			return Arrays.toString(trgWords) + " :: " + Arrays.toString(features);
		}
	}

	private final HashNgramMap<PhraseTableValues> map;

	private final WordIndexer<String> wordIndexer;

	public static MosesPhraseTable readFromFile(final String file) {
		final StringWordIndexer stringWordIndexer = new StringWordIndexer();
		final MosesPhraseTableReaderCallback<String> callback = new MosesPhraseTableReaderCallback<String>(stringWordIndexer);
		new MosesPhraseTableReader<String>(file, stringWordIndexer).parse(callback);
		return new MosesPhraseTable(callback.getMap(), stringWordIndexer);

	}

	private MosesPhraseTable(final HashNgramMap<PhraseTableValues> map, final WordIndexer<String> wordIndexer) {
		this.map = map;
		this.wordIndexer = wordIndexer;
	}

	public List<TargetSideTranslation> getTranslations(final int[] src, final int startPos, final int endPos) {
		final long offsetForNgram = map.getOffsetForNgramInModel(src, startPos, endPos);
		if (offsetForNgram < 0) return Collections.emptyList();
		final TargetTranslationsValues scratch = new PhraseTableValueContainer.TargetTranslationsValues();
		map.getValues().getFromOffset(offsetForNgram, endPos - startPos - 1, scratch);
		final List<TargetSideTranslation> ret = new ArrayList<TargetSideTranslation>();
		for (int i = 0; i < scratch.targetTranslationOffsets.length; ++i) {
			final FeaturePhraseTableValues features = new PhraseTableValueContainer.FeaturePhraseTableValues(null);
			final long currOffset = scratch.targetTranslationOffsets[i];
			final int currOrder = scratch.targetTranslationOrders[i];
			map.getValues().getFromOffset(currOffset, currOrder, features);
			if (features.features == null) {
				Logger.warn("Should probably fix");
				continue;
			}
			final TargetSideTranslation tr = new TargetSideTranslation();
			tr.features = Arrays.copyOf(features.features, features.features.length);
			int sepIndex = 0;
			final int[] srcAndTrg = map.getNgramForOffset(currOffset, currOrder);
			for (; sepIndex < srcAndTrg.length; ++sepIndex) {
				if (srcAndTrg[sepIndex] == ((PhraseTableValueContainer) map.getValues()).getSeparatorWord()) {
					break;
				}
			}
			tr.trgWords = Arrays.copyOfRange(srcAndTrg, sepIndex + 1, srcAndTrg.length);
			assert tr.trgWords.length > 0;
			ret.add(tr);
		}
		return ret;

	}

	public WordIndexer<String> getWordIndexer() {
		return wordIndexer;
	}

}
