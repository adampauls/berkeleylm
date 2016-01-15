package edu.berkeley.nlp.lm.phrasetable;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.io.LmReaderCallback;
import edu.berkeley.nlp.lm.map.HashNgramMap;

/**
 * Class for representing phrase tables efficiently in memory.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class MosesPhraseTableReaderCallback<W> implements LmReaderCallback<PhraseTableCounts>
{

	private final HashNgramMap<PhraseTableValueContainer.PhraseTableValues> phrases;

	public MosesPhraseTableReaderCallback(final WordIndexer<W> wordIndexer) {
		final PhraseTableValueContainer values = new PhraseTableValueContainer(wordIndexer.getOrAddIndexFromString(MosesPhraseTableReader.SEP_WORD), 5);
		phrases = HashNgramMap.createExplicitWordHashNgramMap(values, new ConfigOptions(), 20, false);
	}

	@Override
	public void call(final int[] ngram, final int startPos, final int endPos, final PhraseTableCounts value, final String words) {
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
