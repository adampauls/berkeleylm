package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.BackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.ConfigOptions;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.hash.MurmurHash;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;

public class LmReaders
{

	public static BackoffLm<String> readArpaLmFile(final String lmFile) {
		return readArpaLmFile(lmFile, new StringWordIndexer());
	}

	public static <W> BackoffLm<W> readArpaLmFile(final String lmFile, final WordIndexer<W> wordIndexer) {
		return readArpaLmFile(lmFile, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
	}

	/**
	 * Factory method for reading an ARPA lm file.
	 * 
	 * @param <W>
	 * @param opts
	 * @param lmFile
	 * @param lmOrder
	 * @param wordIndexer
	 * @return
	 */
	public static <W> BackoffLm<W> readArpaLmFile(final String lmFile, final WordIndexer<W> wordIndexer, final ConfigOptions opts, final int lmOrder) {

		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPass(lmFile, lmOrder, wordIndexer);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPass(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord);
	}

	/**
	 * Second pass actually builds the lm.
	 * 
	 * @param <W>
	 * @param opts
	 * @param lmFile
	 * @param lmOrder
	 * @param wordIndexer
	 * @param valueAddingCallback
	 * @param numNgramsForEachWord
	 * @return
	 */
	private static <W> BackoffLm<W> secondPass(final ConfigOptions opts, final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord) {
		Logger.startTrack("Pass 2 of 2");
		final ProbBackoffValueContainer values = new ProbBackoffValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, opts.storeSuffixIndexes);
		final NgramMap<ProbBackoffPair> map = new HashNgramMap<ProbBackoffPair>(values, new MurmurHash(), opts, numNgramsForEachWord, opts.storeSuffixIndexes);
		final ARPALmReader<W> arpaLmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		arpaLmReader.parse(new NgramMapAddingCallback<ProbBackoffPair>(map));
		wordIndexer.trimAndLock();
		Logger.endTrack();
		return new BackoffLm<W>(lmOrder, wordIndexer, map, opts);
	}

	/**
	 * First pass over the file collects some statistics which help with memory
	 * allocation
	 * 
	 * @param <W>
	 * @param opts
	 * @param lmFile
	 * @param lmOrder
	 * @param wordIndexer
	 * @return
	 */
	private static <W> FirstPassCallback<ProbBackoffPair> firstPass(final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer) {
		Logger.startTrack("Pass 1 of 2");
		final ARPALmReader<W> arpaLmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = new FirstPassCallback<ProbBackoffPair>();
		arpaLmReader.parse(valueAddingCallback);
		Logger.endTrack();
		return valueAddingCallback;
	}

}
