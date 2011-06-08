package edu.berkeley.nlp.lm.phrasetable;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.io.LmReaderCallback;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap.Entry;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.StrUtils;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

/**
 * Class for representing phrase tables efficiently in memory.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class MosesPhraseTableReaderCallback<W> implements LmReaderCallback<PhraseTableCounts>
{

	private HashNgramMap<PhraseTableValueContainer.PhraseTableValues> phrases;

	public MosesPhraseTableReaderCallback(WordIndexer<W> wordIndexer) {
		final PhraseTableValueContainer values = new PhraseTableValueContainer(wordIndexer.getOrAddIndexFromString(MosesPhraseTableReader.SEP_WORD), 5);
		phrases = HashNgramMap.createExplicitWordHashNgramMap(values, new ConfigOptions(), 20, false);
	}

	@Override
	public void call(int[] ngram, int startPos, int endPos, PhraseTableCounts value, String words) {
		for (int ngramOrder = 0; ngramOrder < endPos - startPos; ++ngramOrder)
			phrases.put(ngram, startPos, startPos + ngramOrder + 1, new PhraseTableValueContainer.TargetTranslationsValues());
		phrases.put(ngram, startPos, endPos, new PhraseTableValueContainer.FeaturePhraseTableValues(value.features));
	}

	@Override
	public void cleanup() {
		phrases.trim();
	}

	public HashNgramMap<PhraseTableValueContainer.PhraseTableValues> getMap() {
		return phrases;
	}

}
