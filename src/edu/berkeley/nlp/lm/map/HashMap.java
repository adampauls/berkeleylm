package edu.berkeley.nlp.lm.map;

import java.util.Iterator;

import edu.berkeley.nlp.lm.array.CustomWidthArray;
import edu.berkeley.nlp.lm.array.LongArray;

interface HashMap
{

	
	public long put(final long key);

	public long getOffset(final long key);

	public double getLoadFactor();

	public long getCapacity();

	public long getKey(long contextOffset);

	public boolean isEmptyKey(long key);

	public long size();

	public Iterable<Long> keys();

	

	public boolean hasContexts(int word);

}