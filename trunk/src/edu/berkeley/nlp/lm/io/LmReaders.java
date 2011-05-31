package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.map.CompressedNgramMap;
import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.values.CompressibleValueContainer;
import edu.berkeley.nlp.lm.values.RankedCountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;
import edu.berkeley.nlp.lm.values.ValueContainer;

public class LmReaders
{

	public static ContextEncodedProbBackoffLm<String> readContextEncodedLmFromArpa(final String lmFile, boolean compress) {
		return readContextEncodedLmFromArpa(lmFile, compress, new StringWordIndexer());
	}

	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedLmFromArpa(final String lmFile, boolean compress, final WordIndexer<W> wordIndexer) {
		return readContextEncodedLmFromArpa(lmFile, compress, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
	}

	/**
	 * Factory method for reading an ARPA lm file.
	 * 
	 * @param <W>
	 * @param opts
	 * @param lmFile
	 * @param lmOrder
	 * @param wordIndexer
	 * @param compress
	 * @return
	 */
	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedLmFromArpa(final String lmFile, boolean compress, final WordIndexer<W> wordIndexer,
		final ConfigOptions opts, final int lmOrder) {

		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer, false);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassContextEncoded(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, compress);
	}

	public static ProbBackoffLm<String> readLmFromArpa(final String lmFile, boolean compress) {
		return readLmFromArpa(lmFile, compress, new StringWordIndexer());
	}

	public static <W> ProbBackoffLm<W> readLmFromArpa(final String lmFile, final boolean compress, final WordIndexer<W> wordIndexer) {
		return readLmFromArpa(lmFile, compress, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
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
	public static <W> ProbBackoffLm<W> readLmFromArpa(final String lmFile, boolean compress, final WordIndexer<W> wordIndexer, final ConfigOptions opts,
		final int lmOrder) {

		final boolean reverse = true;
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer, reverse);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPass(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, reverse, compress);
	}

	public static StupidBackoffLm<String> readLmFromGoogleNgramDir(final String dir, boolean compress) {
		return readLmFromGoogleNgramDir(dir, compress, new StringWordIndexer(), new ConfigOptions());
	}

	public static <W> StupidBackoffLm<W> readLmFromGoogleNgramDir(final String dir, boolean compress, final WordIndexer<W> wordIndexer, final ConfigOptions opts) {
		final FirstPassCallback<LongRef> valueAddingCallback = firstPassGoogle(dir, wordIndexer, opts);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassGoogle(opts, dir, wordIndexer, valueAddingCallback, numNgramsForEachWord, compress);
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
		final WordIndexer<W> wordIndexer, final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord,
		final boolean compress) {
		final boolean contextEncoded = true;
		final boolean reversed = false;
		final NgramMap<ProbBackoffPair> map = buildMapArpa(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded,
			reversed, compress);
		return new ContextEncodedProbBackoffLm<W>(lmOrder, wordIndexer, (ContextEncodedNgramMap<ProbBackoffPair>) map, opts);
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
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord, final boolean reversed, boolean compress) {
		final boolean contextEncoded = false;
		final NgramMap<ProbBackoffPair> map = buildMapArpa(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded,
			reversed, compress);
		return new ProbBackoffLm<W>(lmOrder, wordIndexer, map, opts);
	}

	private static <W> StupidBackoffLm<W> secondPassGoogle(final ConfigOptions opts, final String dir, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<LongRef> valueAddingCallback, final LongArray[] numNgramsForEachWord, boolean compress) {
		final boolean contextEncoded = false;
		final boolean reversed = true;
		final RankedCountValueContainer values = new RankedCountValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, contextEncoded);
		final GoogleLmReader<W> lmReader = new GoogleLmReader<W>(dir, wordIndexer, opts);
		final NgramMap<LongRef> map = buildMapCommon(opts, wordIndexer, numNgramsForEachWord, reversed, lmReader, values, compress);
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
	private static <W> NgramMap<ProbBackoffPair> buildMapArpa(final ConfigOptions opts, final String lmFile, final int lmOrder,
		final WordIndexer<W> wordIndexer, final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord,
		final boolean contextEncoded, final boolean reversed, final boolean compress) {
		final ARPALmReader<W> lmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final ProbBackoffValueContainer values = new ProbBackoffValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, contextEncoded);
		if (contextEncoded && compress) { throw new IllegalArgumentException("Compression is not supported by context-encoded LMs. Please set "
			+ ConfigOptions.class.getSimpleName() + ".compress to false"); }
		final NgramMap<ProbBackoffPair> map = buildMapCommon(opts, wordIndexer, numNgramsForEachWord, reversed, lmReader, values, compress);
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
		final LongArray[] numNgramsForEachWord, final boolean reversed, final LmReader<V> lmReader, final ValueContainer<V> values, final boolean compress) {
		Logger.startTrack("Pass 2 of 2");
		final NgramMap<V> map = compress ? new CompressedNgramMap<V>((CompressibleValueContainer<V>) values, opts) : HashNgramMap.createImplicitWordHashNgramMap(values, opts, numNgramsForEachWord, reversed);

		lmReader.parse(new NgramMapAddingCallback<V>(map));
		wordIndexer.trimAndLock();
		Logger.endTrack();
		return map;
	}

	private static <W> FirstPassCallback<ProbBackoffPair> firstPassArpa(final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer,
		final boolean reverse) {
		final ARPALmReader<W> arpaLmReader = new ARPALmReader<W>(lmFile, wordIndexer, lmOrder);
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassCommon(arpaLmReader, reverse);
		return valueAddingCallback;
	}

	private static <W> FirstPassCallback<LongRef> firstPassGoogle(final String rootDir, final WordIndexer<W> wordIndexer, final ConfigOptions opts) {
		final GoogleLmReader<W> arpaLmReader = new GoogleLmReader<W>(rootDir, wordIndexer, opts);
		final boolean reverse = true;
		final FirstPassCallback<LongRef> valueAddingCallback = firstPassCommon(arpaLmReader, reverse);
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
	private static <V extends Comparable<V>> FirstPassCallback<V> firstPassCommon(final LmReader<V> arpaLmReader, final boolean reverse) {
		Logger.startTrack("Pass 1 of 2");
		final FirstPassCallback<V> valueAddingCallback = new FirstPassCallback<V>(reverse);
		arpaLmReader.parse(valueAddingCallback);
		Logger.endTrack();
		return valueAddingCallback;
	}

}
