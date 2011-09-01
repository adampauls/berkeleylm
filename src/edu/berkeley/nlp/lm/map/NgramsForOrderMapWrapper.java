package edu.berkeley.nlp.lm.map;

import java.util.AbstractMap;
import java.util.AbstractSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Wraps an NgramMap as a Java Map, but only ngrams of a particular order. This
 * collection is read-only. It is also uses a lot inefficient boxing and
 * unboxing.
 * 
 * @author adampauls
 * 
 * @param <W>
 * @param <V>
 */
public class NgramsForOrderMapWrapper<W, V> extends AbstractMap<List<W>, V>
{

	private final NgramMap<V> map;

	private final int ngramOrder;

	private final WordIndexer<W> wordIndexer;

	private final NgramsForOrderIterableWrapper<W, V> iterableWrapper;

	/**
	 * 
	 * @param map
	 * @param ngramOrder
	 *            0-based, i.e. 0 means unigrams
	 */
	public NgramsForOrderMapWrapper(final NgramMap<V> map, final WordIndexer<W> wordIndexer, final int ngramOrder) {
		this.map = map;
		this.ngramOrder = ngramOrder;
		this.wordIndexer = wordIndexer;
		iterableWrapper = new NgramsForOrderIterableWrapper<W, V>(map, wordIndexer, ngramOrder);
	}

	@Override
	public V get(final Object arg0) {
		if (!(arg0 instanceof List)) return null;
		@SuppressWarnings("unchecked")
		final List<W> l = (List<W>) arg0;

		if (l.size() != ngramOrder + 1) return null;
		final int[] ngram = WordIndexer.StaticMethods.toArray(wordIndexer, l);

		return getForArray(ngram);

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
				return iterableWrapper.iterator();
			}

			@Override
			public int size() {
				final long size = iterableWrapper.size();
				if (size > Integer.MAX_VALUE)
					Logger.warn(NgramsForOrderMapWrapper.class.getSimpleName() + " doesn't like maps with size greater than Integer.MAX_VALUE");
				return (int) size;
			}
		};
	}

	/**
	 * @param scratch
	 * @param ngram
	 */
	private V getForArray(final int[] ngram) {
		return map.get(ngram, 0, ngram.length);
		//		long probContext = 0L;
		//		int probContextOrder = -1;
		//		final V scratch = map.getValues().getScratchValue();
		//		final NgramMap<V> localMap = map;
		//		final int endPos_ = ngram.length;
		//		final int startPos_ = 0;
		//		localMap.
		//		for (int i = endPos_ - 1; i >= startPos_; --i) {
		//			probContext = localMap.getValueAndOffset(probContext, probContextOrder, ngram[i], scratch);
		//			if (probContext < 0) return null;
		//			probContextOrder++;
		//		}
		//		return scratch;
	}

}
