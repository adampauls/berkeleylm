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
	private LongArray uncompressedKeys;

	private long numKeys;

	public long add(final long key) {
		uncompressedKeys.addWithFixedCapacity(key);
		return uncompressedKeys.size();
	}

	public long size() {
		return uncompressedKeys == null ? numKeys : uncompressedKeys.size();
	}

	public void init(final long l) {
		uncompressedKeys = LongArray.StaticMethods.newLongArray(Long.MAX_VALUE, l, l);
	}

	public void trim() {
		uncompressedKeys.trim();
	}

	public void clearUncompressedKeys() {
		numKeys = uncompressedKeys.size();
		uncompressedKeys = null;
	}

	public LongArray getUncompressedKeys() {
		return uncompressedKeys;
	}

}