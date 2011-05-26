package edu.berkeley.nlp.lm.io;


public interface LmReader<V>
{
	public void parse(final LmReaderCallback<V> callback_);

}
