package edu.berkeley.nlp.lm.map;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Wraps an NgramMap as an Iterable, so it is easy to iterate over the n-grams
 * and associated values. Using this interface is a little inefficient due to
 * the boxing and temporary object allocation necessary to conform to Java's
 * interfaces.
 * 
 * @author adampauls
 * 
 * @param <V>
 * @param <W>
 */
public class NgramIterableWrapper<W, V> implements Iterable<java.util.Map.Entry<List<W>, V>>
{

	private final NgramsForOrderIterableWrapper<W, V>[] ngramsForOrder;

	public NgramIterableWrapper(final NgramMap<V> map, final WordIndexer<W> wordIndexer) {
		this(map, wordIndexer, map.getMaxNgramOrder());
	}

	/**
	 * 
	 * @param map
	 * @param wordIndexer
	 * @param maxOrder
	 *            this is 1-based (i.e. 1 means keep unigrams but not bigrams)
	 */
	public NgramIterableWrapper(final NgramMap<V> map, final WordIndexer<W> wordIndexer, final int maxOrder) {
		@SuppressWarnings("unchecked")
		final NgramsForOrderIterableWrapper<W, V>[] maps = new NgramsForOrderIterableWrapper[maxOrder];
		ngramsForOrder = maps;
		for (int ngramOrder = 0; ngramOrder < maxOrder; ++ngramOrder) {
			ngramsForOrder[ngramOrder] = new NgramsForOrderIterableWrapper<W, V>(map, wordIndexer, ngramOrder);
		}
	}

	@Override
	public Iterator<Entry<List<W>, V>> iterator() {
		final Iterators.Transform<NgramsForOrderIterableWrapper<W, V>, Iterator<java.util.Map.Entry<List<W>, V>>> transform = new Iterators.Transform<NgramsForOrderIterableWrapper<W, V>, Iterator<java.util.Map.Entry<List<W>, V>>>(
			Arrays.asList(ngramsForOrder).iterator())
		{

			@Override
			protected Iterator<java.util.Map.Entry<List<W>, V>> transform(final NgramsForOrderIterableWrapper<W, V> next) {
				return next.iterator();
			}
		};
		return new Iterators.IteratorIterator<Map.Entry<List<W>, V>>(transform);

	}

	public long size() {
		long size = 0;
		for (final NgramsForOrderIterableWrapper<W, V> map : ngramsForOrder) {
			size += map.size();
		}
		if (size > Integer.MAX_VALUE) Logger.warn(NgramMapWrapper.class.getSimpleName() + " doesn't like maps with size greater than Integer.MAX_VALUE");
		return (int) size;
	}
}
