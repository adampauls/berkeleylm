package edu.berkeley.nlp.lm.map;

import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.map.NgramMap.Entry;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

public class JavaMapWrapper<W, T> extends AbstractMap<List<W>, T>
{
	private final NgramMap<T> map;

	private final int ngramOrder;

	private final WordIndexer<W> wordIndexer;

	private final IterableWrapper<T, W> iterableWrapper;

	/**
	 * 
	 * @param map
	 * @param ngramOrder
	 *            0-based, i.e. 0 means unigrams
	 */
	public JavaMapWrapper(NgramMap<T> map, WordIndexer<W> wordIndexer, int ngramOrder) {
		this.map = map;
		this.ngramOrder = ngramOrder;
		this.wordIndexer = wordIndexer;
		iterableWrapper = new IterableWrapper<T, W>(map, wordIndexer, ngramOrder);
	}

	@Override
	public T get(Object arg0) {
		if (!(arg0 instanceof List)) return null;
		@SuppressWarnings("unchecked")
		List<W> l = (List<W>) arg0;

		if (l.size() != ngramOrder + 1) return null;
		int[] ngram = WordIndexer.StaticMethods.toArray(wordIndexer, l);

		return getForArray(ngram);

	}

	/**
	 * @param scratch
	 * @param ngram
	 */
	private T getForArray(int[] ngram) {
		long probContext = 0L;
		int probContextOrder = -1;
		T scratch = map.getValues().getScratchValue();
		NgramMap<T> localMap = map;
		int endPos_ = ngram.length;
		int startPos_ = 0;
		for (int i = endPos_ - 1; i >= startPos_; --i) {
			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);
			if (probContext < 0) return null;
			probContextOrder++;
		}
		return scratch;
	}

	@Override
	public Set<java.util.Map.Entry<List<W>, T>> entrySet() {
		return new AbstractSet<java.util.Map.Entry<List<W>, T>>()
		{

			@Override
			public Iterator<java.util.Map.Entry<List<W>, T>> iterator() {

				return iterableWrapper.iterator();
			}

			@Override
			public int size() {
				// warning: if there are more than 2^31 n-grams for this order, this will cause trouble.
				return (int) iterableWrapper.size();
			}
		};
	}

}
