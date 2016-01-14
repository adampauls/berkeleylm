package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.util.Arrays;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;

/**
 * Reads in n-gram count collections in the format that the Google n-grams Web1T
 * corpus comes in.
 * 
 * @author adampauls
 * 
 */
public class GoogleLmReader<W> implements LmReader<LongRef, NgramOrderedLmReaderCallback<LongRef>>
{

	public int getLmOrder() {
		return lmOrder;
	}

	private static final String START_SYMBOL = "<S>";

	private static final String END_SYMBOL = "</S>";

	private static final String UNK_SYMBOL = "<UNK>";

	private static final String sortedVocabFile = "vocab_cs.gz";

	private final File[] ngramDirectories;
	
	private final int lmOrder;

	private final WordIndexer<W> wordIndexer;

	public GoogleLmReader(final String rootDir, final WordIndexer<W> wordIndexer, @SuppressWarnings("unused") final ConfigOptions opts) {
		this.wordIndexer = wordIndexer;
		ngramDirectories = new File(rootDir).listFiles(new FilenameFilter()
		{

			@Override
			public boolean accept(final File dir, final String name) {
				return name.endsWith("gms");
			}
		});
		Arrays.sort(ngramDirectories);
		lmOrder = ngramDirectories.length;
	}

	@Override
	public void parse(final NgramOrderedLmReaderCallback<LongRef> callback) {
		int ngramOrder = 0;
		for (final File ngramDir : ngramDirectories) {
			final int ngramOrder_ = ngramOrder;
			final String regex = (ngramOrder_ + 1) + "gm-\\d+(.gz)?";
			final File[] ngramFiles = ngramDir.listFiles(new FilenameFilter()
			{

				@Override
				public boolean accept(final File dir, final String name) {
					return ngramOrder_ == 0 ? name.equals(sortedVocabFile) : name.matches(regex);
				}
			});
			if (ngramOrder == 0) {
				if (ngramFiles.length != 1) throw new RuntimeException("Could not find expected vocab file " + sortedVocabFile);
				final String sortedVocabPath = ngramFiles[0].getPath();
				addToIndexer(wordIndexer, sortedVocabPath);
			} else if (ngramFiles.length == 0) {
				Logger.warn("Did not find any files matching expected regex " + regex);
			}
			Arrays.sort(ngramFiles);
			Logger.startTrack("Reading ngrams of order " + (ngramOrder_ + 1));
			for (final File ngramFile_ : ngramFiles) {
				final File ngramFile = ngramFile_;
				Logger.startTrack("Reading ngrams from file " + ngramFile);
				try {
					int k = 0;
					for (String line : Iterators.able(IOUtils.lineIterator(ngramFile.getPath()))) {
						if (k % 10000 == 0) Logger.logs("Line " + k);
						k++;
						line = line.trim();
						try {
							parseLine(line, ngramOrder, callback);
						} catch (Throwable e) {
							throw new RuntimeException("Could not parse line " + k + " '" + line + "' from file " + ngramFile + "\n", e);
						}
					}
				} catch (final IOException e) {
					throw new RuntimeException("Could not read file " + ngramFile + "\n", e);

				}
				Logger.endTrack();
			}

			Logger.endTrack();
			callback.handleNgramOrderFinished(++ngramOrder);

		}
		callback.cleanup();

	}

	/**
	 * @param callback
	 * @param ngramOrder
	 * @param line
	 * @return
	 */
	private void parseLine(final String line, final int ngramOrder, final NgramOrderedLmReaderCallback<LongRef> callback) {
		final int tabIndex = line.indexOf('\t');

		int spaceIndex = 0;
		final int[] ngram = new int[ngramOrder + 1];
		final String words = line.substring(0, tabIndex);
		for (int i = 0;; ++i) {
			int nextIndex = line.indexOf(' ', spaceIndex);
			if (nextIndex < 0) nextIndex = words.length();
			final String word = words.substring(spaceIndex, nextIndex);
			ngram[i] = wordIndexer.getOrAddIndexFromString(word);

			if (nextIndex == words.length()) break;
			spaceIndex = nextIndex + 1;
		}
		final long count = Long.parseLong(line.substring(tabIndex + 1));
		callback.call(ngram, 0, ngram.length, new LongRef(count), words);
	}

	/**
	 * @param sortedVocabPath
	 */
	public static <W> void addToIndexer(final WordIndexer<W> wordIndexer, final String sortedVocabPath) {
		if (!(new File(sortedVocabPath).getName().equals(sortedVocabFile))) {
			Logger.warn("You have specified that " + sortedVocabPath + " is the count-sorted vocab file for Google n-grams, but it is usually named "
				+ sortedVocabFile);
		}
		try {
			for (final String line : Iterators.able(IOUtils.lineIterator(sortedVocabPath))) {
				final String[] parts = line.split("\t");
				final String word = parts[0];
				wordIndexer.getOrAddIndexFromString(word);
			}
		} catch (final NumberFormatException e) {
			throw new RuntimeException(e);

		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
		addSpecialSymbols(wordIndexer);
	}

	/**
	 * 
	 */
	static <W> void addSpecialSymbols(final WordIndexer<W> wordIndexer) {
		wordIndexer.setStartSymbol(wordIndexer.getWord(wordIndexer.getOrAddIndexFromString(START_SYMBOL)));
		wordIndexer.setEndSymbol(wordIndexer.getWord(wordIndexer.getOrAddIndexFromString(END_SYMBOL)));
		wordIndexer.setUnkSymbol(wordIndexer.getWord(wordIndexer.getOrAddIndexFromString(UNK_SYMBOL)));
	}

}
