package edu.berkeley.nlp.lm.io;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.collections.Counter;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Reader callback which adds n-grams to an NgramMap
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type
 */
public final class ValueAddingCallback<V extends Comparable<V>> implements LmReaderCallback<V>
{

	int warnCount = 0;

	Counter<V> valueCounter;

	private Indexer<V> valueIndexer;

	public ValueAddingCallback() {
		this.valueCounter = new Counter<V>();
	}

	@Override
	public void call(final int[] ngram, final V v, final String words) {
		valueCounter.incrementCount(v, 1);
	}

	@Override
	public void handleNgramOrderFinished(final int order) {
	}

	@Override
	public void cleanup() {
		Logger.startTrack("Cleaning up values");
		valueIndexer = new Indexer<V>();
		for (Entry<V, Double> entry : valueCounter.getEntriesSortedByDecreasingCount()) {
			valueIndexer.add(entry.getKey());
		}
		Logger.logss("Found " + valueIndexer.size() + " unique counts");

		valueCounter = null;
		Logger.endTrack();

	}

	public Indexer<V> getIndexer() {
		return valueIndexer;

	}

	@Override
	public void initWithLengths(final List<Long> numNGrams) {
	}

	@Override
	public boolean ignoreNgrams() {
		return true;
	}
}