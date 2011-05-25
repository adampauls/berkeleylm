package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.ConfigOptions;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.hash.MurmurHash;
import edu.berkeley.nlp.lm.values.CountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;
import edu.berkeley.nlp.lm.values.ValueContainer;

public class LmReaders
{

	public static ContextEncodedProbBackoffLm<String> readContextEncodedLmFromArpa(final String lmFile) {
		return readContextEncodedLmFromArpa(lmFile, new StringWordIndexer());
	}

	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedLmFromArpa(final String lmFile, final WordIndexer<W> wordIndexer) {
		return readContextEncodedLmFromArpa(lmFile, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
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
	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedLmFromArpa(final String lmFile, final WordIndexer<W> wordIndexer,
		final ConfigOptions opts, final int lmOrder) {

		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassContextEncoded(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord);
	}

	public static ProbBackoffLm<String> readLmFromArpa(final String lmFile) {
		return readLmFromArpa(lmFile, new StringWordIndexer());
	}

	public static <W> ProbBackoffLm<W> readLmFromArpa(final String lmFile, final WordIndexer<W> wordIndexer) {
		return readLmFromArpa(lmFile, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
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
	public static <W> ProbBackoffLm<W> readLmFromArpa(final String lmFile, final WordIndexer<W> wordIndexer, final ConfigOptions opts, final int lmOrder) {

		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPass(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord);
	}

	public static StupidBackoffLm<String> readLmFromGoogleNgramDir(final String dir) {
		return readLmFromGoogleNgramDir(dir, new StringWordIndexer(), new ConfigOptions());
	}

	public static <W> StupidBackoffLm<W> readLmFromGoogleNgramDir(final String dir, final WordIndexer<W> wordIndexer, final ConfigOptions opts) {
		final FirstPassCallback<LongRef> valueAddingCallback = firstPassGoogle(dir, wordIndexer, opts);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassGoogle(opts, dir, wordIndexer, valueAddingCallback, numNgramsForEachWord);
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
	private static <W> ContextEncodedProbBackoffLm<W> secondPassContextEncoded(final ConfigOptions opts, final String lmFile, final int lmOrder,
		final WordIndexer<W> wordIndexer, final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord) {
		final boolean contextEncoded = true;
		final boolean reversed = false;
		final NgramMap<ProbBackoffPair> map = buildMap(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded, reversed);
		return new ContextEncodedProbBackoffLm<W>(lmOrder, wordIndexer, map, opts);
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
	private static <W> ProbBackoffLm<W> secondPass(final ConfigOptions opts, final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord) {
		final boolean contextEncoded = false;
		final boolean reversed = true;
		final NgramMap<ProbBackoffPair> map = buildMap(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded, reversed);
		return new ProbBackoffLm<W>(lmOrder, wordIndexer, map, opts);
	}

	private static <W> StupidBackoffLm<W> secondPassGoogle(final ConfigOptions opts, final String dir, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<LongRef> valueAddingCallback, final LongArray[] numNgramsForEachWord) {
		final boolean contextEncoded = false;
		final boolean reversed = true;
		final CountValueContainer values = new CountValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, contextEncoded);
		final GoogleLmReader<W> lmReader = new GoogleLmReader<W>(dir, wordIndexer, opts);
		final NgramMap<LongRef> map = buildMapCommon(opts, wordIndexer, numNgramsForEachWord, reversed, lmReader, values);
		return new StupidBackoffLm<W>(numNgramsForEachWord.length, wordIndexer, map, opts);
	}

	/**
	 * @param <W>
	 * @param opts
	 * @param lmFile
	 * @param lmOrder
	 * @param wordIndexer
	 * @param valueAddingCallback
	 * @param numNgramsForEachWord
	 * @param contextEncoded
	 * @param reversed
	 * @return
	 */
	private static <W> NgramMap<ProbBackoffPair> buildMap(final ConfigOptions opts, final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord, final boolean contextEncoded,
		final boolean reversed) {
		final ARPALmReader<W> lmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final ProbBackoffValueContainer values = new ProbBackoffValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, contextEncoded);
		final NgramMap<ProbBackoffPair> map = buildMapCommon(opts, wordIndexer, numNgramsForEachWord, reversed, lmReader, values);
		return map;
	}

	/**
	 * @param <W>
	 * @param opts
	 * @param wordIndexer
	 * @param valueAddingCallback
	 * @param numNgramsForEachWord
	 * @param contextEncoded
	 * @param reversed
	 * @param lmReader
	 * @return
	 */
	private static <W, V extends Comparable<V>> NgramMap<V> buildMapCommon(final ConfigOptions opts, final WordIndexer<W> wordIndexer,
		final LongArray[] numNgramsForEachWord, final boolean reversed, final LmReader<V> lmReader, final ValueContainer<V> values) {
		Logger.startTrack("Pass 2 of 2");
		final NgramMap<V> map = new HashNgramMap<V>(values, new MurmurHash(), opts, numNgramsForEachWord, reversed);

		lmReader.parse(new NgramMapAddingCallback<V>(map));
		wordIndexer.trimAndLock();
		Logger.endTrack();
		return map;
	}

	private static <W> FirstPassCallback<ProbBackoffPair> firstPassArpa(final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer) {
		final ARPALmReader<W> arpaLmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassCommon(arpaLmReader);
		return valueAddingCallback;
	}

	private static <W> FirstPassCallback<LongRef> firstPassGoogle(final String rootDir, final WordIndexer<W> wordIndexer, ConfigOptions opts) {
		final GoogleLmReader<W> arpaLmReader = new GoogleLmReader<W>(rootDir, wordIndexer, opts);
		final FirstPassCallback<LongRef> valueAddingCallback = firstPassCommon(arpaLmReader);
		return valueAddingCallback;
	}

	/**
	 * First pass over the file collects some statistics which help with memory
	 * allocation
	 * 
	 * @param <W>
	 * @param arpaLmReader
	 * @return
	 */
	private static <V extends Comparable<V>> FirstPassCallback<V> firstPassCommon(final LmReader<V> arpaLmReader) {
		Logger.startTrack("Pass 1 of 2");
		final FirstPassCallback<V> valueAddingCallback = new FirstPassCallback<V>();
		arpaLmReader.parse(valueAddingCallback);
		Logger.endTrack();
		return valueAddingCallback;
	}

}
