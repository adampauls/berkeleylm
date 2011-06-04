package edu.berkeley.nlp.lm.io;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.array.LongArray;
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
public final class FirstPassCallback<V extends Comparable<V>> implements ArpaLmReaderCallback<V>
{

	private Counter<V> valueCounter;

	private Indexer<V> valueIndexer;

	private LongArray[] numNgramsForEachWord;

	private long[] numNgramsForOrder;

	private final boolean reverse;

	private int maxNgramOrder = 0;

	public FirstPassCallback(final boolean reverse) {
		this.reverse = reverse;
		this.valueCounter = new Counter<V>();
	}

	@Override
	public void call(final int[] ngram, int startPos, int endPos, final V v, final String words) {
		maxNgramOrder = Math.max(endPos - startPos, maxNgramOrder);
		final int ngramOrder = endPos - startPos - 1;
		allocatedNumNgramArrayIfNecessary(ngramOrder);
		allocatedNumNgramForOrderArrayIfNecessary(ngramOrder);
		valueCounter.incrementCount(v, 1);
		final LongArray ngramOrderCounts = numNgramsForEachWord[ngramOrder];
		final int word = reverse ? ngram[startPos] : ngram[ngramOrder];
		ngramOrderCounts.incrementCount(word, 1);
		numNgramsForOrder[ngramOrder]++;
		//		if (word >= ngramOrderCounts.size()) {
		//
		//			ngramOrderCounts.setAndGrowIfNeeded(word, 1);
		//		} else {
		//			ngramOrderCounts.set(word, ngramOrderCounts.get(word) + 1);
		//		}

	}

	@Override
	public void handleNgramOrderFinished(final int order) {
	}

	@Override
	public void cleanup() {
		Logger.startTrack("Cleaning up values");
		valueIndexer = new Indexer<V>();
		for (final Entry<V, Double> entry : valueCounter.getEntriesSortedByDecreasingCount()) {
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
		maxNgramOrder = numNGrams.size();
		final long numWords = numNGrams.get(0);
		numNgramsForEachWord = new LongArray[numNGrams.size()];
		for (int ngramOrder = 0; ngramOrder < numNgramsForEachWord.length; ++ngramOrder) {
			numNgramsForEachWord[ngramOrder] = LongArray.StaticMethods.newLongArray(numNGrams.get(ngramOrder), numWords, numWords);
		}
	}

	public LongArray[] getNumNgramsForEachWord() {
		return Arrays.copyOf(numNgramsForEachWord, maxNgramOrder);
	}

	public long[] getNumNgramsForEachOrder() {
		return Arrays.copyOf(numNgramsForOrder, maxNgramOrder);
	}

	/**
	 * @param startPos
	 * @param endPos
	 * @return
	 */
	private int allocatedNumNgramArrayIfNecessary(int ngramOrder) {
		if (numNgramsForEachWord == null) {
			numNgramsForEachWord = new LongArray[5];
		}

		if (ngramOrder >= numNgramsForEachWord.length) {
			numNgramsForEachWord = Arrays.copyOf(numNgramsForEachWord, numNgramsForEachWord.length * 2);
		}
		if (numNgramsForEachWord[ngramOrder] == null) {
			numNgramsForEachWord[ngramOrder] = LongArray.StaticMethods.newLongArray(Integer.MAX_VALUE, Integer.MAX_VALUE);
		}
		return ngramOrder;
	}

	private int allocatedNumNgramForOrderArrayIfNecessary(int ngramOrder) {
		if (numNgramsForOrder == null) {
			numNgramsForOrder = new long[5];
		}
		if (ngramOrder >= numNgramsForOrder.length) {
			numNgramsForOrder = Arrays.copyOf(numNgramsForOrder, numNgramsForOrder.length * 2);
		}

		return ngramOrder;
	}

}