package edu.berkeley.nlp.lm.map;

import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Wraps an NgramMap as a Java Map, with ngrams of all orders mixed together.
 * This collection is read-only. It is also uses a lot inefficient boxing and
 * unboxing.
 * 
 * @author adampauls
 * 
 * @param <W>
 * @param <V>
 */
public class NgramMapWrapper<W, V> extends AbstractMap<List<W>, V>
{

	private final NgramsForOrderMapWrapper<W, V>[] ngramsForOrder;

	private final WordIndexer<W> wordIndexer;

	private final NgramMap<V> ngramMap;

	public NgramMapWrapper(final NgramMap<V> map, final WordIndexer<W> wordIndexer) {
		this(map, wordIndexer, map.getMaxNgramOrder());
	}

	/**
	 * 
	 * @param map
	 * @param wordIndexer
	 * @param maxOrder
	 *            this is 1-based (i.e. 1 means keep unigrams but not bigrams)
	 */
	public NgramMapWrapper(final NgramMap<V> map, final WordIndexer<W> wordIndexer, final int maxOrder) {
		@SuppressWarnings("unchecked")
		final NgramsForOrderMapWrapper<W, V>[] maps = new NgramsForOrderMapWrapper[maxOrder];
		ngramsForOrder = maps;
		for (int ngramOrder = 0; ngramOrder < maxOrder; ++ngramOrder) {
			ngramsForOrder[ngramOrder] = new NgramsForOrderMapWrapper<W, V>(map, wordIndexer, ngramOrder);
		}
		this.wordIndexer = wordIndexer;
		this.ngramMap = map;
	}

	@Override
	public V get(final Object arg0) {
		if (!(arg0 instanceof List)) return null;
		@SuppressWarnings("unchecked")
		final List<W> l = (List<W>) arg0;

		if (l.size() > ngramsForOrder.length) return null;
		return ngramsForOrder[l.size() - 1].get(l);

	}

	@Override
	public boolean containsKey(final Object key) {
		return get(key) != null;
	}

	@Override
	public Set<java.util.Map.Entry<List<W>, V>> entrySet() {
		return new AbstractSet<java.util.Map.Entry<List<W>, V>>()
		{

			@Override
			public Iterator<java.util.Map.Entry<List<W>, V>> iterator() {
				final Iterators.Transform<NgramsForOrderMapWrapper<W, V>, Iterator<java.util.Map.Entry<List<W>, V>>> transform = new Iterators.Transform<NgramsForOrderMapWrapper<W, V>, Iterator<java.util.Map.Entry<List<W>, V>>>(
					Arrays.asList(ngramsForOrder).iterator())
				{

					@Override
					protected Iterator<java.util.Map.Entry<List<W>, V>> transform(final NgramsForOrderMapWrapper<W, V> next) {
						return next.entrySet().iterator();
					}
				};
				return new Iterators.IteratorIterator<Map.Entry<List<W>, V>>(transform);
			}

			@Override
			public int size() {
				if (longSize() > Integer.MAX_VALUE)
					Logger.warn(NgramMapWrapper.class.getSimpleName() + " doesn't like maps with size greater than Integer.MAX_VALUE");

				return (int) longSize();
			}

		};
	}

	/**
	 * 
	 * @param ngramOrder
	 *            0-based (0 means unigrams)
	 * @return
	 */
	public Map<List<W>, V> getMapForOrder(final int ngramOrder) {
		return ngramsForOrder[ngramOrder];
	}

	/**
	 * @return
	 */
	public long longSize() {
		long size = 0;
		for (final NgramsForOrderMapWrapper<W, V> map : ngramsForOrder) {
			size += map.size();
		}
		return size;
	}

	public WordIndexer<W> getWordIndexer() {
		return wordIndexer;
	}

	public NgramMap<V> getNgramMap() {
		return ngramMap;
	}
}
