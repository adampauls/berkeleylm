package edu.berkeley.nlp.lm.map;

import java.util.List;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ValueContainer;

public interface NgramMap<T>
{

	public long put(int[] ngram, T val);

	public void handleNgramsFinished(int justFinishedOrder);

	public void trim();

	public void initWithLengths(List<Long> numNGrams);

	public void getValue(int[] ngram, int startPos, int endPos, @OutputParameter LmContextInfo contextOutput, @OutputParameter T outputVal);

	public ValueContainer<T> getValues();

	public LmContextInfo getOffsetForNgram(int[] ngram, int startPos, int endPos);

}
