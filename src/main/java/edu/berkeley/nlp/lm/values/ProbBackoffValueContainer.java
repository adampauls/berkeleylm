package edu.berkeley.nlp.lm.values;

public interface ProbBackoffValueContainer extends ValueContainer<ProbBackoffPair>
{

	public abstract float getProb(final int ngramOrder, final long index);

	public abstract float getBackoff(final int ngramOrder, final long index);

	public abstract ProbBackoffPair getScratchValue();

	public abstract long getSuffixOffset(final long index, final int ngramOrder);

}