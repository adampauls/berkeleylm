package edu.berkeley.nlp.lm.phrasetable;

import java.io.IOException;
import java.util.Arrays;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.io.IOUtils;
import edu.berkeley.nlp.lm.io.LmReader;
import edu.berkeley.nlp.lm.io.LmReaderCallback;
import edu.berkeley.nlp.lm.util.Logger;

public class MosesPhraseTableReader<W> implements LmReader<PhraseTableCounts, MosesPhraseTableReaderCallback<W>>
{

	static final String SEP_WORD = "<<sep>>";

	private final WordIndexer<W> wordIndexer;

	private final String file;

	public MosesPhraseTableReader(final String file, final WordIndexer<W> wordIndexer) {
		this.file = file;
		this.wordIndexer = wordIndexer;

	}

	@Override
	public void parse(final MosesPhraseTableReaderCallback<W> callback) {
		readFromFiles(callback);
	}

	private void readFromFiles(final LmReaderCallback<PhraseTableCounts> callback) {
		Logger.startTrack("Reading from file " + file);
		try {
			final Iterable<String> allLinesIterator = Iterators.able(IOUtils.lineIterator(file));
			countPhrases(allLinesIterator, callback);
		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
		Logger.endTrack();

	}

	/**
	 * @param <W>
	 * @param wordIndexer
	 * @param maxOrder
	 * @param allLinesIterator
	 * @param callback
	 * @param ngrams
	 * @return
	 */
	private void countPhrases(final Iterable<String> allLinesIterator, final LmReaderCallback<PhraseTableCounts> callback) {
		long numLines = 0;

		for (final String line : allLinesIterator) {
			if (numLines % 10000 == 0) Logger.logs("On line " + numLines);
			numLines++;
			final String[] parts = line.trim().split("\\|\\|\\|");
			if (parts.length != 5 && parts.length != 3) throw new IllegalArgumentException("Bad Moses phrase table file line " + line);
			assert (parts.length == 3 || parts.length == 5);
			// ingore alignments if they exist
			if (parts.length == 5) parts[2] = parts[4];

			final String[] src = parts[0].trim().split("\\s+");
			final int[] srcInts = WordIndexer.StaticMethods.toArrayFromStrings(wordIndexer, Arrays.asList(src));
			final String[] trg = parts[1].trim().split("\\s+");
			final int[] trgInts = WordIndexer.StaticMethods.toArrayFromStrings(wordIndexer, Arrays.asList(trg));

			final int sepIndex = wordIndexer.getOrAddIndexFromString(SEP_WORD);
			final String[] featStrings = parts[2].trim().split("\\s+");
			final float[] features = new float[featStrings.length];
			// we skip the last feature since it is the bias, and is always the same.
			for (int i = 0; i < featStrings.length - 1; i++) {
				try {
					final Float val = Float.parseFloat(featStrings[i]);
					if (val.isInfinite() || val.isNaN()) {
						Logger.warn("Non-finite feature: " + featStrings[i]);
						continue;
					}

					features[i] = (float) -Math.log(val);
				} catch (final NumberFormatException n) {
					throw new RuntimeException("Bad Moses phrase table file line: " + line);
				}
			}

			final int[] concat = new int[srcInts.length + trgInts.length + 1];
			System.arraycopy(srcInts, 0, concat, 0, srcInts.length);
			concat[srcInts.length] = sepIndex;
			System.arraycopy(trgInts, 0, concat, srcInts.length + 1, trgInts.length);
			callback.call(concat, 0, concat.length, new PhraseTableCounts(features), line);

		}
		callback.cleanup();
	}

}
