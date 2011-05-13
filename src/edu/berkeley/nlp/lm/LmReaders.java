package edu.berkeley.nlp.lm;

import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.io.ARPALmReader;
import edu.berkeley.nlp.lm.io.NgramMapAddingCallback;
import edu.berkeley.nlp.lm.io.ValueAddingCallback;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.NgramMapOpts;
import edu.berkeley.nlp.lm.util.hash.MurmurHash;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

public class LmReaders
{

	public static <W> KatzBackoffLm<W> readArpaLmFile(final NgramMapOpts opts, final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer) {

		return readArpaLmFile(opts, lmFile, lmOrder, wordIndexer, true);
	}

	public static <W> KatzBackoffLm<W> readArpaLmFile(final NgramMapOpts opts, final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer,
		final boolean lockIndexer) {
		final ARPALmReader<W> arpaLmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final ValueAddingCallback<ProbBackoffPair> valueAddingCallback = new ValueAddingCallback<ProbBackoffPair>(opts);
		arpaLmReader.parse(valueAddingCallback);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		final ProbBackoffValueContainer values = new ProbBackoffValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, opts.storePrefixIndexes);
		final NgramMap<ProbBackoffPair> map = new HashNgramMap<ProbBackoffPair>(values, new MurmurHash(), opts, numNgramsForEachWord, 100000);

		new ARPALmReader<W>(lmFile, wordIndexer, lmOrder).parse(new NgramMapAddingCallback<ProbBackoffPair>(map));
		if (lockIndexer) wordIndexer.trimAndLock();

		return new KatzBackoffLm<W>(lmOrder, wordIndexer, map, opts);
	}

}
