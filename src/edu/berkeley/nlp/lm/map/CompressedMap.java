package edu.berkeley.nlp.lm.map;

import java.io.Serializable;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.util.Annotations.PrintMemoryCount;

class CompressedMap implements Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	@PrintMemoryCount
	LongArray compressedKeys;

	@PrintMemoryCount
	LongArray keys;

	public long add(final long key) {
		keys.add(key);
		return keys.size();
	}

	public long size() {
		return keys.size();
	}

	public void init(final long l) {
		keys = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, l, l);
	}

	public void trim() {
		keys.trim();
	}

}