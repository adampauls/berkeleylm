package edu.berkeley.nlp.lm.collections;

import java.util.Iterator;

import edu.berkeley.nlp.lm.util.Pair;

/**
 * Utilities for dealing with Iterators
 * 
 * @author adampauls
 * 
 */
public class Iterators
{

	/**
	 * Wraps an Iterator as an Iterable
	 * 
	 * @param <T>
	 * @param it
	 * @return
	 */
	public static <T> Iterable<T> able(final Iterator<T> it) {
		return new Iterable<T>()
		{
			boolean used = false;

			@Override
			public Iterator<T> iterator() {
				if (used) throw new RuntimeException("One use iterable");
				used = true;
				return it;
			}
		};
	}

	public static <T> Iterable<T> flatten(final Iterator<Iterator<T>> iters) {
		return Iterators.able(new IteratorIterator<T>(iters));
	}

	/**
	 * Wraps a two-level iteration scenario in an iterator. Each key of the keys
	 * iterator returns an iterator (via the factory) over T's.
	 * <p>
	 * The IteratorIterator loops through the iterator associated with each key
	 * until all the keys are used up.
	 */
	public static class IteratorIterator<T> implements Iterator<T>
	{
		Iterator<T> current = null;

		private final Iterator<Iterator<T>> iters;

		public IteratorIterator(final Iterator<Iterator<T>> iters) {
			this.iters = iters;
			current = getNextIterator();
		}

		private Iterator<T> getNextIterator() {
			Iterator<T> next = null;
			while (next == null) {
				if (!iters.hasNext()) break;
				next = iters.next();
				if (!next.hasNext()) next = null;
			}
			return next;
		}

		@Override
		public boolean hasNext() {
			return current != null;
		}

		@Override
		public T next() {
			final T next = current.next();
			if (!current.hasNext()) current = getNextIterator();
			return next;
		}

		@Override
		public void remove() {
			throw new UnsupportedOperationException();
		}

	}

	/**
	 * Wraps a base iterator with a transformation function.
	 */
	public static abstract class Transform<S, T> implements Iterator<T>
	{

		private final Iterator<S> base;

		public Transform(final Iterator<S> base) {
			this.base = base;
		}

		@Override
		public boolean hasNext() {
			return base.hasNext();
		}

		@Override
		public T next() {
			return transform(base.next());
		}

		protected abstract T transform(S next);

		@Override
		public void remove() {
			base.remove();
		}

	}

	public static <S, T> Iterator<Pair<S, T>> zip(final Iterator<S> s, final Iterator<T> t) {
		return new Iterator<Pair<S, T>>()
		{
			@Override
			public boolean hasNext() {
				return s.hasNext() && t.hasNext();
			}

			@Override
			public Pair<S, T> next() {
				return Pair.newPair(s.next(), t.next());
			}

			@Override
			public void remove() {
				throw new UnsupportedOperationException();
			}
		};
	}

}
