package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.util.List;

import edu.berkeley.nlp.lm.ArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ArrayEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.array.LongArray;
import edu.berkeley.nlp.lm.cache.ArrayEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.cache.ContextEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.map.AbstractNgramMap;
import edu.berkeley.nlp.lm.map.CompressedNgramMap;
import edu.berkeley.nlp.lm.map.ContextEncodedNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.map.NgramMapWrapper;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.values.CompressibleValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.ProbBackoffValueContainer;
import edu.berkeley.nlp.lm.values.RankedCountValueContainer;
import edu.berkeley.nlp.lm.values.ValueContainer;

/**
 * This class contains a number of static methods for reading/writing/estimating
 * n-gram language models. Since most uses of this software will use this class,
 * I will use this space to document the software as a whole.
 * <p>
 * This software provides three main pieces of functionality: <br>
 * (a) estimation of a language models from text inputs <br>
 * (b) data structures for efficiently storing large collections of n-grams in
 * memory <br>
 * (c) an API for efficient querying language models derived from n-gram
 * collections. Most of the techniques in the paper are described in
 * "Faster and Smaller N-gram Language Models" (Pauls and Klein 2011).
 * <p>
 * This software supports the estimation of two types of language models:
 * Kneser-Ney language models (Kneser and Ney, 1995) and Stupid Backoff language
 * models (Brants et al. 2007). Kneser-Ney language models can be estimated from
 * raw text by called
 * {@link #createKneserNeyLmFromTextFiles(List, WordIndexer, int, File)}. This
 * can also be done from the command-line by calling <code>main()</code> in
 * {@link MakeKneserNeyArpaFromText}. See the <code>examples</code> folder for a
 * script which demonstrates its use. A Stupid Backoff language model can be
 * read from a directory containing n-gram counts in the format used by Google's
 * Web1T corpus by calling {@link #readLmFromGoogleNgramDir(String, boolean)}.
 * Note that this software does not (yet) support building Google count
 * directories from raw text, though this can be done using SRILM.
 * <p>
 * Loading/estimating language models from text files can be very slow. This
 * software can use Java's built-in serialization to build language model
 * binaries which are both smaller and faster to load.
 * {@link MakeLmBinaryFromArpa} and {@link MakeLmBinaryFromGoogle} provide
 * <code>main()</code> methods for doing this. See the <code>examples</code>
 * folder for scripts which demonstrate their use.
 * <p>
 * Language models can be read into memory from ARPA formats using
 * {@link #readArrayEncodedLmFromArpa(String, boolean)} and
 * {@link #readContextEncodedLmFromArpa(String)}. The "array encoding" versus
 * "context encoding" distinction is discussed in Section 4.2 of Pauls and Klein
 * (2011). Again, since loading language models from textual representations can
 * be very slow, they can be read from binaries using
 * {@link #readLmBinary(String)}. The interfaces for these language models can
 * be found in {@link ArrayEncodedNgramLanguageModel} and
 * {@link ContextEncodedNgramLanguageModel}. For examples of these interfaces in
 * action, you can have a look at {@link PerplexityTest}.
 * <p>
 * We implement the HASH,HASH+SCROLL, and COMPRESSED language model
 * representations described in Pauls and Klein (2011) in this release. The
 * SORTED implementation may be added later. See {@link HashNgramMap} and
 * {@link CompressedNgramMap} for the implementations of the HASH and COMPRESSED
 * representations.
 * <p>
 * To speed up queries, you can wrap language models with caches (
 * {@link ContextEncodedCachingLmWrapper} and
 * {@link ArrayEncodedCachingLmWrapper}). These caches are described in section
 * 4.1 of Pauls and Klein (2011). You should more or less always use these
 * caches, since they are faster and have modest memory requirements. Note,
 * however, that the caches are <b>not</b> synchronized. The only threadsafe way
 * to use them is to have a separate caching wrapper for each separate decoding
 * thread (though they can of course all wrap the same underlying LM).
 * <p>
 * This software also support a java Map wrapper around an n-gram collection.
 * You can read a map wrapper using
 * {@link #readNgramMapFromGoogleNgramDir(String, boolean, WordIndexer)}.
 * 
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

	public static ArrayEncodedProbBackoffLm<String> readArrayEncodedLmFromArpa(final String lmFile, final boolean compress) {
		return readArrayEncodedLmFromArpa(lmFile, compress, new StringWordIndexer());
	}

	public static <W> ArrayEncodedProbBackoffLm<W> readArrayEncodedLmFromArpa(final String lmFile, final boolean compress, final WordIndexer<W> wordIndexer) {
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
	public static <W> ArrayEncodedProbBackoffLm<W> readArrayEncodedLmFromArpa(final String lmFile, final boolean compress, final WordIndexer<W> wordIndexer,
		final ConfigOptions opts, final int lmOrder) {

		final boolean reverse = true;
		final FirstPassCallback<ProbBackoffPair> valueAddingCallback = firstPassArpa(lmFile, lmOrder, wordIndexer, reverse);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassArrayEncoded(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, reverse, compress);
	}

	public static NgramMapWrapper<String, LongRef> readNgramMapFromGoogleNgramDir(final String dir, final boolean compress) {
		return readNgramMapFromGoogleNgramDir(dir, compress, new StringWordIndexer());
	}

	public static <W> NgramMapWrapper<W, LongRef> readNgramMapFromGoogleNgramDir(final String dir, final boolean compress, final WordIndexer<W> wordIndexer) {
		final StupidBackoffLm<W> lm = readLmFromGoogleNgramDir(dir, compress, wordIndexer, new ConfigOptions());
		return new NgramMapWrapper<W, LongRef>(lm.getNgramMap(), lm.getWordIndexer());
	}

	public static NgramMapWrapper<String, LongRef> readNgramMapFromBinary(final String binary, final String vocabFile) {
		return readNgramMapFromBinary(binary, vocabFile, new StringWordIndexer());
	}

	/**
	 * 
	 * @param sortedVocabFile
	 *            should be the vocab_cs.gz file from the Google n-gram corpus.
	 * @return
	 */
	public static <W> NgramMapWrapper<W, LongRef> readNgramMapFromBinary(final String binary, final String sortedVocabFile, final WordIndexer<W> wordIndexer) {
		GoogleLmReader.addToIndexer(wordIndexer, sortedVocabFile);
		wordIndexer.trimAndLock();
		@SuppressWarnings("unchecked")
		final NgramMap<LongRef> map = (NgramMap<LongRef>) IOUtils.readObjFileHard(binary);
		return new NgramMapWrapper<W, LongRef>(map, wordIndexer);
	}

	public static StupidBackoffLm<String> readLmFromGoogleNgramDir(final String dir, final boolean compress) {
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
	public static <W> StupidBackoffLm<W> readLmFromGoogleNgramDir(final String dir, final boolean compress, final WordIndexer<W> wordIndexer,
		final ConfigOptions opts) {
		final FirstPassCallback<LongRef> valueAddingCallback = firstPassGoogle(dir, wordIndexer, opts);
		final LongArray[] numNgramsForEachWord = valueAddingCallback.getNumNgramsForEachWord();
		return secondPassGoogle(opts, dir, wordIndexer, valueAddingCallback, numNgramsForEachWord, compress);
	}

	/**
	 * Builds a context-encoded LM from raw text. This call first builds and
	 * writes a (temporary) ARPA file by calling
	 * {@link #createKneserNeyLmFromTextFiles(List, WordIndexer, int, File)},
	 * and the reads the resulting file. Since the temp file can be quite large,
	 * it is important that the temp directory used by java (
	 * <code>java.io.tmpdir</code>).
	 * 
	 * @param <W>
	 * @param files
	 * @param wordIndexer
	 * @param lmOrder
	 * @param opts
	 * @return
	 */
	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedKneserNeyLmFromTextFile(final List<File> files, final WordIndexer<W> wordIndexer,
		final int lmOrder, final ConfigOptions opts) {
		final File tmpFile = getTempFile();
		return readContextEncodedKneserNeyLmFromTextFile(files, wordIndexer, lmOrder, opts, tmpFile);
	}

	/**
	 * Builds an array-encoded LM from raw text. This call first builds and
	 * writes a (temporary) ARPA file by calling
	 * {@link #createKneserNeyLmFromTextFiles(List, WordIndexer, int, File)},
	 * and the reads the resulting file. Since the temp file can be quite large,
	 * it is important that the temp directory used by java (
	 * <code>java.io.tmpdir</code>).
	 * 
	 * @param <W>
	 * @param files
	 * @param wordIndexer
	 * @param lmOrder
	 * @param opts
	 * @return
	 */
	public static <W> ArrayEncodedProbBackoffLm<W> readKneserNeyLmFromTextFile(final List<File> files, final WordIndexer<W> wordIndexer, final int lmOrder,
		final ConfigOptions opts, final boolean compress) {
		final File tmpFile = getTempFile();
		return readKneserNeyLmFromTextFile(files, wordIndexer, lmOrder, compress, opts, tmpFile);
	}

	public static <W> ContextEncodedProbBackoffLm<W> readContextEncodedKneserNeyLmFromTextFile(final List<File> files, final WordIndexer<W> wordIndexer,
		final int lmOrder, final ConfigOptions opts, final File tmpFile) {
		createKneserNeyLmFromTextFiles(files, wordIndexer, lmOrder, tmpFile, opts);
		return readContextEncodedLmFromArpa(tmpFile.getPath(), wordIndexer, opts, lmOrder);
	}

	public static <W> ArrayEncodedProbBackoffLm<W> readKneserNeyLmFromTextFile(final List<File> files, final WordIndexer<W> wordIndexer, final int lmOrder,
		final boolean compress, final ConfigOptions opts, final File tmpFile) {
		createKneserNeyLmFromTextFiles(files, wordIndexer, lmOrder, tmpFile, opts);
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
	public static <W> void createKneserNeyLmFromTextFiles(final List<File> files, final WordIndexer<W> wordIndexer, final int lmOrder,
		final File arpaOutputFile, final ConfigOptions opts) {
		final TextReader<W> reader = new TextReader<W>(files, wordIndexer, lmOrder);
		reader.parse(new KneserNeyLmReaderCallback<W>(arpaOutputFile, wordIndexer, lmOrder, opts));
	}

	public static StupidBackoffLm<String> readGoogleLmBinary(final String file, final String sortedVocabFile) {
		return readGoogleLmBinary(file, new StringWordIndexer(), sortedVocabFile);
	}

	/**
	 * Reads in a binarized
	 * 
	 * @param <W>
	 * @param file
	 * @param wordIndexer
	 * @param sortedVocabFile
	 * @return
	 */
	public static <W> StupidBackoffLm<W> readGoogleLmBinary(final String file, final WordIndexer<W> wordIndexer, final String sortedVocabFile) {
		GoogleLmReader.addToIndexer(wordIndexer, sortedVocabFile);
		wordIndexer.trimAndLock();
		@SuppressWarnings("unchecked")
		final NgramMap<LongRef> map = (NgramMap<LongRef>) IOUtils.readObjFileHard(file);
		return new StupidBackoffLm<W>(map.getMaxNgramOrder(), wordIndexer, map, new ConfigOptions());
	}

	/**
	 * Reads a binary file representing an LM. These will need to be cast down
	 * to either {@link ContextEncodedNgramLanguageModel} or
	 * {@link ArrayEncodedNgramLanguageModel} to be useful.
	 */
	public static <W> NgramLanguageModel<W> readLmBinary(final String file) {
		@SuppressWarnings("unchecked")
		final NgramLanguageModel<W> lm = (NgramLanguageModel<W>) IOUtils.readObjFileHard(file);
		return lm;
	}

	/**
	 * Writes a binary file representing the LM using the built-in
	 * serialization. These binaries should load much faster than ARPA files.
	 * 
	 * @param <W>
	 * @param lm
	 * @param file
	 */
	public static <W> void writeLmBinary(final NgramLanguageModel<W> lm, final String file) {
		IOUtils.writeObjFileHard(file, lm);
	}

	/**
	 * @return
	 */
	private static File getTempFile() {
		try {
			final File tmpFile = File.createTempFile("berkeleylm", "arpa");
			tmpFile.deleteOnExit();
			return tmpFile;
		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
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
		return new ContextEncodedProbBackoffLm<W>(map.getMaxNgramOrder(), wordIndexer, (HashNgramMap<ProbBackoffPair>) map, opts);
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
	private static <W> ArrayEncodedProbBackoffLm<W> secondPassArrayEncoded(final ConfigOptions opts, final String lmFile, final int lmOrder,
		final WordIndexer<W> wordIndexer, final FirstPassCallback<ProbBackoffPair> valueAddingCallback, final LongArray[] numNgramsForEachWord,
		final boolean reversed, final boolean compress) {
		final boolean contextEncoded = false;
		final NgramMap<ProbBackoffPair> map = buildMapArpa(opts, lmFile, lmOrder, wordIndexer, valueAddingCallback, numNgramsForEachWord, contextEncoded,
			reversed, compress);
		return new ArrayEncodedProbBackoffLm<W>(map.getMaxNgramOrder(), wordIndexer, map, opts);
	}

	private static <W> StupidBackoffLm<W> secondPassGoogle(final ConfigOptions opts, final String dir, final WordIndexer<W> wordIndexer,
		final FirstPassCallback<LongRef> valueAddingCallback, final LongArray[] numNgramsForEachWord, final boolean compress) {
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
		final ArpaLmReader<W> lmReader = new ArpaLmReader<W>(lmFile, wordIndexer, lmOrder);
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
		Logger.startTrack("Adding n-grams");
		NgramMap<V> map = createNgramMap(opts, numNgramsForEachWord, numNgramsForEachOrder, reversed, values, compress);

		final NgramMapAddingCallback<V> ngramMapAddingCallback = new NgramMapAddingCallback<V>(map, null);
		lmReader.parse(ngramMapAddingCallback);
		wordIndexer.trimAndLock();
		Logger.endTrack();
		final List<int[]> failures = ngramMapAddingCallback.getFailures();
		if (!failures.isEmpty()) {
			Logger.startTrack("Some missing suffixes of prefixes were found, doing another pass to add n-grams");
			for (final int[] failure : failures) {
				final int ngramOrder = failure.length - 1;
				final int headWord = failure[reversed ? 0 : ngramOrder];
				numNgramsForEachOrder[ngramOrder]++;
				numNgramsForEachWord[ngramOrder].incrementCount(headWord, 1);
			}
			map = createNgramMap(opts, numNgramsForEachWord, numNgramsForEachOrder, reversed, values.createFreshValues(), compress);
			lmReader.parse(new NgramMapAddingCallback<V>(map, failures));
			Logger.endTrack();
		}
		return map;
	}

	/**
	 * @param <V>
	 * @param opts
	 * @param numNgramsForEachWord
	 * @param numNgramsForEachOrder
	 * @param reversed
	 * @param values
	 * @param compress
	 * @return
	 */
	private static <V> AbstractNgramMap<V> createNgramMap(final ConfigOptions opts, final LongArray[] numNgramsForEachWord, final long[] numNgramsForEachOrder,
		final boolean reversed, final ValueContainer<V> values, final boolean compress) {
		return compress ? new CompressedNgramMap<V>((CompressibleValueContainer<V>) values, numNgramsForEachOrder, opts) : HashNgramMap
			.createImplicitWordHashNgramMap(values, opts, numNgramsForEachWord, reversed);
	}

	private static <W> FirstPassCallback<ProbBackoffPair> firstPassArpa(final String lmFile, final int lmOrder, final WordIndexer<W> wordIndexer,
		final boolean reverse) {
		final ArpaLmReader<W> arpaLmReader = new ArpaLmReader<W>(lmFile, wordIndexer, lmOrder);
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
		Logger.startTrack("Counting values");
		final FirstPassCallback<V> valueAddingCallback = new FirstPassCallback<V>(reverse);
		arpaLmReader.parse(valueAddingCallback);
		Logger.endTrack();
		return valueAddingCallback;
	}

}
