package edu.berkeley.nlp.lm.array;

public interface LongArray
{

	public abstract void set(long pos, long val);

	public abstract void setAndGrowIfNeeded(long pos, long val);

	public abstract long get(long pos);

	public abstract void trim();

	public abstract long size();

	public abstract boolean add(long val);

	public abstract void trimToSize(long size);

	public abstract void fill(long l, long initialCapacity);

}