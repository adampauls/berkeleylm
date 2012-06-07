package edu.berkeley.nlp.lm.io;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.collections.Counter;
import edu.berkeley.nlp.lm.collections.Indexer;
import edu.berkeley.nlp.lm.collections.LongToIntHashMap;
import edu.berkeley.nlp.lm.collections.LongRepresentable;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Reader callback which adds n-grams to an NgramMap
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type
 */
public final class FirstPassCallback<V extends LongRepresentable<V>> implements ArpaLmReaderCallback<V>
{

	private LongToIntHashMap valueCounter;

	private LongArray[] numNgramsForEachWord;

	private long[] numNgramsForOrder;

	private final boolean reverse;

	private int maxNgramOrder = 0;

	public FirstPassCallback(final boolean reverse) {
		this.reverse = reverse;
		this.valueCounter = new LongToIntHashMap();
	}

	@Override
	public void call(final int[] ngram, final int startPos, final int endPos, final V v, final String words) {
		maxNgramOrder = Math.max(endPos - startPos, maxNgramOrder);
		final int ngramOrder = endPos - startPos - 1;
		allocatedNumNgramArrayIfNecessary(ngramOrder);
		allocatedNumNgramForOrderArrayIfNecessary(ngramOrder);
		valueCounter.incrementCount(v.asLong(), 1);
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

		Logger.logss("Found " + valueCounter.size() + " unique counts");

		Logger.endTrack();

	}

	public LongToIntHashMap getValueCounter() {
		return valueCounter;

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
	private int allocatedNumNgramArrayIfNecessary(final int ngramOrder) {
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

	private int allocatedNumNgramForOrderArrayIfNecessary(final int ngramOrder) {
		if (numNgramsForOrder == null) {
			numNgramsForOrder = new long[5];
		}
		if (ngramOrder >= numNgramsForOrder.length) {
			numNgramsForOrder = Arrays.copyOf(numNgramsForOrder, numNgramsForOrder.length * 2);
		}

		return ngramOrder;
	}

	@Override
	public void handleNgramOrderStarted(int order) {

	}

}