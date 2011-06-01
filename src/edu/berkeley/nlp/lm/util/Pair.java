package edu.berkeley.nlp.lm.util;

import java.io.Serializable;

/**
 * A generic-typed pair of objects.
 * 
 * @author Dan Klein
 */
public class Pair<F, S> implements Serializable
{
	static final long serialVersionUID = 42;

	F first;

	S second;

	public F getFirst() {
		return first;
	}

	public S getSecond() {
		return second;
	}

	@Override
	public String toString() {
		return "(" + getFirst() + ", " + getSecond() + ")";
	}

	public Pair(final F first, final S second) {
		this.first = first;
		this.second = second;
	}

	public static <S, T> Pair<S, T> newPair(final S first, final T second) {
		return new Pair<S, T>(first, second);
	}
}
