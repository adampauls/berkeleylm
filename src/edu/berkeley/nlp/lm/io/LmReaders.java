package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.KatzBackoffLm;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.NgramMapOpts;
import edu.berkeley.nlp.lm.util.hash.MurmurHash;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

public class LmReaders
{

	public static <W> KatzBackoffLm<W> readArpaLmFile(final NgramMapOpts opts, final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer) {

		final ARPALmReader<W> arpaLmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final ValueAddingCallback<ProbBackoffPair> valueAddingCallback = new ValueAddingCallback<ProbBackoffPair>(opts);
		arpaLmReader.parse(valueAddingCallback);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		final ProbBackoffValueContainer values = new ProbBackoffValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, opts.storePrefixIndexes);
		final NgramMap<ProbBackoffPair> map = new HashNgramMap<ProbBackoffPair>(values, new MurmurHash(), opts, numNgramsForEachWord);

		new ARPALmReader<W>(lmFile, wordIndexer, lmOrder).parse(new NgramMapAddingCallback<ProbBackoffPair>(map));
		wordIndexer.trimAndLock();

		return new KatzBackoffLm<W>(lmOrder, wordIndexer, map, opts);
	}

}
