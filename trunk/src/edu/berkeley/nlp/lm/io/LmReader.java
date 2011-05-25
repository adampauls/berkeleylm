package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.values.ProbBackoffPair;

public interface LmReader<V>
{
	public void parse(final LmReaderCallback<V> callback_);

}
