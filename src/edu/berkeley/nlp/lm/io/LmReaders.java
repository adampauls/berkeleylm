package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.NgramLanguageModel;
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

/**
 * Factory methods for reading/writing language models.
 * 
 * @author adampauls
 * 
 */
public class LmReaders
{

	public static ContextEncodedProbBackoffLm<String> readContextEncodedLmFromArpa(final String lmFile) {
		return readContextEncodedLmFromArpa(lmFile, new StringWordIndexer());
	}

	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedLmFromArpa(final String lmFile, final WordIndexer<W> wordIndexer) {
		return readContextEncodedLmFromArpa(lmFile, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
	}

	/**
	 * Reads a context-encoded language model from an ARPA lm file.
	 * Context-encoded language models allow faster queries, but require an
	 * extra 4-bytes of storage per n-gram for the suffix offsets (as compared
	 * to array-encoded language models).
	 * 
	 * @param <W>
	 * @param lmFile
	 * @param compress
	 * @param wordIndexer
	 * @param opts
	 * @param lmOrder
	 * @return
	 */
	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedLmFromArpa(final String lmFile, final WordIndexer<W> wordIndexer,
		final ConfigOptions opts, final int lmOrder) {

		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer, false);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassContextEncoded(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord);
	}

	public static ProbBackoffLm<String> readArrayEncodedLmFromArpa(final String lmFile, boolean compress) {
		return readArrayEncodedLmFromArpa(lmFile, compress, new StringWordIndexer());
	}

	public static <W> ProbBackoffLm<W> readArrayEncodedLmFromArpa(final String lmFile, final boolean compress, final WordIndexer<W> wordIndexer) {
		return readArrayEncodedLmFromArpa(lmFile, compress, wordIndexer, new ConfigOptions(), Integer.MAX_VALUE);
	}

	/**
	 * Reads an array-encoded language model from an ARPA lm file.
	 * 
	 * @param <W>
	 * @param lmFile
	 * @param compress
	 *            Compress the LM using block compression. This LM should be
	 *            smaller but slower.
	 * @param wordIndexer
	 * @param opts
	 * @param lmOrder
	 * @return
	 */
	public static <W> ProbBackoffLm<W> readArrayEncodedLmFromArpa(final String lmFile, boolean compress, final WordIndexer<W> wordIndexer,
		final ConfigOptions opts, final int lmOrder) {

		final boolean reverse = true;
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer, reverse);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassArrayEncoded(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, reverse, compress);
	}

	public static StupidBackoffLm<String> readLmFromGoogleNgramDir(final String dir, boolean compress) {
		return readLmFromGoogleNgramDir(dir, compress, new StringWordIndexer(), new ConfigOptions());
	}

	/**
	 * Reads a stupid backoff lm from a directory with n-gram counts in the
	 * format used by Google n-grams.
	 * 
	 * @param <W>
	 * @param dir
	 * @param compress
	 * @param wordIndexer
	 * @param opts
	 * @return
	 */
	public static <W> StupidBackoffLm<W> readLmFromGoogleNgramDir(final String dir, boolean compress, final WordIndexer<W> wordIndexer, final ConfigOptions opts) {
		final FirstPassCallback<LongRef> valueAddingCallback = firstPassGoogle(dir, wordIndexer, opts);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassGoogle(opts, dir, wordIndexer, valueAddingCallback, numNgramsForEachWord, compress);
	}

	/**
	 * Builds a context-encoded LM from raw text. This call first builds and writes a (temporary) ARPA file by calling  {@link #createKneserNeyLmFromTextFiles(List, WordIndexer, int, File)},
	 * and the reads the resulting file. Since the temp file can be quite large, it is important that the
	 * temp directory used by java (<code>java.io.tmpdir</code>). 
	 * @param <W>
	 * @param files
	 * @param wordIndexer
	 * @param lmOrder
	 * @param opts
	 * @return
	 */
	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedKneserNeyLmFromTextFile(List<File> files, final WordIndexer<W> wordIndexer,
		final int lmOrder, ConfigOptions opts) {
		File tmpFile = getTempFile();
		return readContextEncodedKneserNeyLmFromTextFile(files, wordIndexer, lmOrder, opts, tmpFile);
	}

	/**
	 * Builds an array-encoded LM from raw text. This call first builds and writes a (temporary) ARPA file by calling  {@link #createKneserNeyLmFromTextFiles(List, WordIndexer, int, File)},
	 * and the reads the resulting file. Since the temp file can be quite large, it is important that the
	 * temp directory used by java (<code>java.io.tmpdir</code>). 
	 * @param <W>
	 * @param files
	 * @param wordIndexer
	 * @param lmOrder
	 * @param opts
	 * @return
	 */
	public static <W> ProbBackoffLm<W> readKneserNeyLmFromTextFile(List<File> files, final WordIndexer<W> wordIndexer, final int lmOrder, ConfigOptions opts,
		boolean compress) {
		File tmpFile = getTempFile();
		return readKneserNeyLmFromTextFile(files, wordIndexer, lmOrder, compress, opts, tmpFile);
	}

	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedKneserNeyLmFromTextFile(List<File> files, final WordIndexer<W> wordIndexer,
		final int lmOrder, ConfigOptions opts, File tmpFile) {
		createKneserNeyLmFromTextFiles(files, wordIndexer, lmOrder, tmpFile);
		return readContextEncodedLmFromArpa(tmpFile.getPath(), wordIndexer, opts, lmOrder);
	}

	public static <W> ProbBackoffLm<W> readKneserNeyLmFromTextFile(List<File> files, final WordIndexer<W> wordIndexer, final int lmOrder, boolean compress,
		ConfigOptions opts, File tmpFile) {
		createKneserNeyLmFromTextFiles(files, wordIndexer, lmOrder, tmpFile);
		return readArrayEncodedLmFromArpa(tmpFile.getPath(), compress, wordIndexer, opts, lmOrder);
	}

	/**
	 * Estimates a Kneser-Ney language model from raw text, and writes a file
	 * (in ARPA format)
	 * 
	 * @param <W>
	 * @param files
	 *            Files of raw text (new-line separated).
	 * @param wordIndexer
	 * @param lmOrder
	 * @param arpaOutputFile
	 */
	public static <W> void createKneserNeyLmFromTextFiles(List<File> files, final WordIndexer<W> wordIndexer, final int lmOrder, File arpaOutputFile) {
		final KneserNeyFromTextReader<W> reader = new KneserNeyFromTextReader<W>(files, wordIndexer, lmOrder);
		reader.parse(new KneserNeyLmReaderCallback<W>(arpaOutputFile, wordIndexer, lmOrder));
	}

	@SuppressWarnings("unchecked")
	public static <W> NgramLanguageModel<W> readLmBinary(final String file) {
		return (NgramLanguageModel<W>) IOUtils.readObjFileHard(file);
	}

	public static <W> void writeLmBinary(NgramLanguageModel<W> lm, String file) {
		IOUtils.writeObjFileHard(file, lm);
	}

	/**
	 * @return
	 */
	private static File getTempFile() {
		File tmpFile;
		try {
			tmpFile = File.createTempFile("berkeleylm", "arpa");
		} catch (IOException e) {
			throw new RuntimeException(e);

		}
		tmpFile.deleteOnExit();
		return tmpFile;
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
		final boolean compress = false;
		final NgramMap<ProbBackoffPair> map = buildMapArpa(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded,
			reversed, compress);
		return new ContextEncodedProbBackoffLm<W>(map.getMaxNgramOrder(), wordIndexer, (ContextEncodedNgramMap<ProbBackoffPair>) map, opts);
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
	private static <W> ProbBackoffLm<W> secondPassArrayEncoded(final ConfigOptions opts, final String lmFile, final int lmOrder,
		final WordIndexer<W> wordIndexer, final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord,
		final boolean reversed, boolean compress) {
		final boolean contextEncoded = false;
		final NgramMap<ProbBackoffPair> map = buildMapArpa(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded,
			reversed, compress);
		return new ProbBackoffLm<W>(map.getMaxNgramOrder(), wordIndexer, map, opts);
	}

	private static <W> StupidBackoffLm<W> secondPassGoogle(final ConfigOptions opts, final String dir, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<LongRef> valueAddingCallback, final LongArray[] numNgramsForEachWord, boolean compress) {
		final boolean contextEncoded = false;
		final boolean reversed = true;
		final RankedCountValueContainer values = new RankedCountValueContainer(valueAddingCallback.getIndexer(), opts.valueRadix, contextEncoded);
		final GoogleLmReader<W> lmReader = new GoogleLmReader<W>(dir, wordIndexer, opts);
		final NgramMap<LongRef> map = buildMapCommon(opts, wordIndexer, numNgramsForEachWord, valueAddingCallback.getNumNgramsForEachOrder(), reversed,
			lmReader, values, compress);
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
		assert !(contextEncoded && compress) : "Compression is not supported by context-encoded LMs";
		final NgramMap<ProbBackoffPair> map = buildMapCommon(opts, wordIndexer, numNgramsForEachWord, valueAddingCallback.getNumNgramsForEachOrder(), reversed,
			lmReader, values, compress);
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
		final LongArray[] numNgramsForEachWord, final long[] numNgramsForEachOrder, final boolean reversed,
		final LmReader<V, ? super NgramMapAddingCallback<V>> lmReader, final ValueContainer<V> values, final boolean compress) {
		Logger.startTrack("Pass 2 of 2");
		final NgramMap<V> map = compress ? new CompressedNgramMap<V>((CompressibleValueContainer<V>) values, numNgramsForEachOrder, opts) : HashNgramMap
			.createImplicitWordHashNgramMap(values, opts, numNgramsForEachWord, reversed);

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
	private static <V extends Comparable<V>> FirstPassCallback<V> firstPassCommon(final LmReader<V, ? super FirstPassCallback<V>> arpaLmReader,
		final boolean reverse) {
		Logger.startTrack("Pass 1 of 2");
		final FirstPassCallback<V> valueAddingCallback = new FirstPassCallback<V>(reverse);
		arpaLmReader.parse(valueAddingCallback);
		Logger.endTrack();
		return valueAddingCallback;
	}

}
