package edu.berkeley.nlp.lm.map;

import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;

/**
 * Wraps an NgramMap as an Iterable, so it is easy to iterate over the n-grams of a particular order. Using this interface
 * is a little inefficient due to the boxing and temporary object allocation necessary to conform to Java's interfaces. 
 * @author adampauls
 *
 * @param <T>
 * @param <W>
 */
public class IterableWrapper<T, W> implements Iterable<java.util.Map.Entry<List<W>, T>>
{

	private final NgramMap<T> map;

	private final int ngramOrder;

	private final WordIndexer<W> wordIndexer;

	/**
	 * 
	 * @param map
	 * @param ngramOrder
	 *            0-based, i.e. 0 means unigrams
	 */
	public IterableWrapper(NgramMap<T> map, WordIndexer<W> wordIndexer, int ngramOrder) {
		this.map = map;
		this.ngramOrder = ngramOrder;
		this.wordIndexer = wordIndexer;
	}

	@Override
	public Iterator<Entry<List<W>, T>> iterator() {
		return new Iterators.Transform<NgramMap.Entry<T>, java.util.Map.Entry<List<W>, T>>(map.getNgramsForOrder(ngramOrder).iterator())
		{

			@Override
			protected Entry<List<W>, T> transform(final edu.berkeley.nlp.lm.map.NgramMap.Entry<T> next) {
				return new java.util.Map.Entry<List<W>, T>()
				{

					@Override
					public List<W> getKey() {
						final List<W> ngram = WordIndexer.StaticMethods.toList(wordIndexer, next.key);
						return ngram;
					}

					@Override
					public T getValue() {
						return next.value;
					}

					@Override
					public T setValue(T arg0) {
						throw new UnsupportedOperationException("Method not yet implemented");
					}
				};
			}
		};

	}

	public long size() {
		return map.getNumNgrams(ngramOrder);
	}
}
