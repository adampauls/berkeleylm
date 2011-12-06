package edu.berkeley.nlp.lm.util;

import java.io.Serializable;

import edu.berkeley.nlp.lm.collections.LongRepresentable;

public class LongRef implements Comparable<LongRef>, Serializable, LongRepresentable<LongRef>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public long value;

	public LongRef(final long value) {
		this.value = value;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + (int) (value ^ (value >>> 32));
		return result;
	}

	@Override
	public boolean equals(final Object obj) {
		if (this == obj) return true;
		if (obj == null) return false;
		if (getClass() != obj.getClass()) return false;
		final LongRef other = (LongRef) obj;
		if (value != other.value) return false;
		return true;
	}

	@Override
	public int compareTo(final LongRef arg0) {
		return Double.compare(value, arg0.value);
	}

	@Override
	public String toString() {
		return "" + value;
	}

	@Override
	public long asLong() {
		return value;
	}

}
