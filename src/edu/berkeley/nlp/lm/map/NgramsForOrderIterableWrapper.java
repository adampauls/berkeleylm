package edu.berkeley.nlp.lm.map;

import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;

/**
 * Wraps an NgramMap as an Iterable, so it is easy to iterate over the n-grams
 * of a particular order. Using this interface is a little inefficient due to
 * the boxing and temporary object allocation necessary to conform to Java's
 * interfaces.
 * 
 * @author adampauls
 * 
 * @param <V>
 * @param <W>
 */
public class NgramsForOrderIterableWrapper<W, V> implements Iterable<java.util.Map.Entry<List<W>, V>>
{

	private final NgramMap<V> map;

	private final int ngramOrder;

	private final WordIndexer<W> wordIndexer;

	/**
	 * 
	 * @param map
	 * @param ngramOrder
	 *            0-based, i.e. 0 means unigrams
	 */
	public NgramsForOrderIterableWrapper(final NgramMap<V> map, final WordIndexer<W> wordIndexer, final int ngramOrder) {
		this.map = map;
		this.ngramOrder = ngramOrder;
		this.wordIndexer = wordIndexer;
	}

	@Override
	public Iterator<Entry<List<W>, V>> iterator() {
		return new Iterators.Transform<NgramMap.Entry<V>, java.util.Map.Entry<List<W>, V>>(map.getNgramsForOrder(ngramOrder).iterator())
		{

			@Override
			protected Entry<List<W>, V> transform(final edu.berkeley.nlp.lm.map.NgramMap.Entry<V> next) {
				return new java.util.Map.Entry<List<W>, V>()
				{

					@Override
					public List<W> getKey() {
						final List<W> ngram = WordIndexer.StaticMethods.toList(wordIndexer, next.key);
						return ngram;
					}

					@Override
					public V getValue() {
						return next.value;
					}

					@Override
					public V setValue(final V arg0) {
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
