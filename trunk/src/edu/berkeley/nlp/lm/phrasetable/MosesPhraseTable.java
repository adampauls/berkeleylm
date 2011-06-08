package edu.berkeley.nlp.lm.phrasetable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.phrasetable.PhraseTableValueContainer.FeaturePhraseTableValues;
import edu.berkeley.nlp.lm.phrasetable.PhraseTableValueContainer.PhraseTableValues;
import edu.berkeley.nlp.lm.phrasetable.PhraseTableValueContainer.TargetTranslationsValues;

/**
 * 
 * Experimental class for reading Moses phrase tables and storing them
 * efficiently in memory using a trie.
 * 
 * @author adampauls
 * 
 */
public class MosesPhraseTable
{

	public static class TargetSideTranslation
	{
		float[] features;

		int[] trgWords;

		public String toString() {
			return Arrays.toString(trgWords) + " :: " + Arrays.toString(features);
		}
	}

	private final HashNgramMap<PhraseTableValues> map;

	private final WordIndexer<String> wordIndexer;

	public static MosesPhraseTable readFromFile(String file) {
		final StringWordIndexer stringWordIndexer = new StringWordIndexer();
		final MosesPhraseTableReaderCallback<String> callback = new MosesPhraseTableReaderCallback<String>(stringWordIndexer);
		new MosesPhraseTableReader<String>(file, stringWordIndexer).parse(callback);
		return new MosesPhraseTable(callback.getMap(), stringWordIndexer);

	}

	private MosesPhraseTable(HashNgramMap<PhraseTableValues> map, WordIndexer<String> wordIndexer) {
		this.map = map;
		this.wordIndexer = wordIndexer;
	}

	public List<TargetSideTranslation> getTranslations(int[] src, int startPos, int endPos) {
		long offsetForNgram = map.getOffsetForNgramInModel(src, startPos, endPos);
		if (offsetForNgram < 0) return Collections.emptyList();
		TargetTranslationsValues scratch = new PhraseTableValueContainer.TargetTranslationsValues();
		map.getValues().getFromOffset(offsetForNgram, endPos - startPos - 1, scratch);
		List<TargetSideTranslation> ret = new ArrayList<TargetSideTranslation>();
		for (int i = 0; i < scratch.targetTranslationOffsets.length; ++i) {
			FeaturePhraseTableValues features = new PhraseTableValueContainer.FeaturePhraseTableValues(null);
			final long currOffset = scratch.targetTranslationOffsets[i];
			final int currOrder = scratch.targetTranslationOrders[i];
			map.getValues().getFromOffset(currOffset, currOrder, features);
			TargetSideTranslation tr = new TargetSideTranslation();
			tr.features = Arrays.copyOf(features.features, features.features.length);
			int sepIndex = 0;
			int[] srcAndTrg = map.getNgramForOffset(currOffset, currOrder);
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
