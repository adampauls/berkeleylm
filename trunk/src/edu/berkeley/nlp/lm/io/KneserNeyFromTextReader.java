package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.HashNgramMap.Entry;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.StrUtils;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

/**
 * Class for producing a Kneser-Ney language model in ARPA format from raw text.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class KneserNeyFromTextReader<W> implements LmReader<Object, LmReaderCallback<Object>>
{
	private int lmOrder;

	private WordIndexer<W> wordIndexer;

	private List<File> inputFiles;

	public KneserNeyFromTextReader(List<File> inputFiles, WordIndexer<W> wordIndexer, int maxOrder) {
		this.inputFiles = inputFiles;
		this.lmOrder = maxOrder;
		this.wordIndexer = wordIndexer;

	}

	/**
	 * Reads newline-separated plain text from inputFiles, and writes an ARPA lm
	 * file to outputFile. If files have a .gz suffix, then they will be
	 * (un)zipped as necessary.
	 * 
	 * @param inputFiles
	 * @param outputFile
	 */
	public void parse(LmReaderCallback<Object> callback) {
		readFromFiles(callback);
	}

	private void readFromFiles(LmReaderCallback<Object> callback) {
		Logger.startTrack("Reading from files " + inputFiles);
		final Iterable<String> allLinesIterator = getLineIterator(inputFiles);

		countNgrams(allLinesIterator, callback);
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
	private void countNgrams(final Iterable<String> allLinesIterator, LmReaderCallback<Object> callback) {
		long numLines = 0;

		for (String line : allLinesIterator) {
			if (numLines % 10000 == 0) Logger.logs("On line " + numLines);
			numLines++;
			final String[] words = line.split(" ");
			int[] sent = new int[words.length + 2];
			sent[0] = wordIndexer.getOrAddIndex(wordIndexer.getStartSymbol());
			sent[sent.length - 1] = wordIndexer.getOrAddIndex(wordIndexer.getEndSymbol());
			for (int i = 0; i < words.length; ++i) {
				sent[i + 1] = wordIndexer.getOrAddIndexFromString(words[i]);
			}
			KneserNeyCounts counts = new KneserNeyCounts();
			for (int ngramOrder = 0; ngramOrder < lmOrder; ++ngramOrder) {
				for (int i = 0; i < sent.length; ++i) {
					if (i - ngramOrder < 0) continue;
					counts.tokenCounts = 1;
					callback.call(sent, i - ngramOrder, i + 1, counts, line);
				}
			}
		}
		callback.cleanup();
	}

	/**
	 * @param files
	 * @return
	 */
	private Iterable<String> getLineIterator(Iterable<File> files) {
		final Iterable<String> allLinesIterator = Iterators.flatten(new Iterators.Transform<File, Iterator<String>>(files.iterator())
		{

			@Override
			protected Iterator<String> transform(File file) {
				try {
					return IOUtils.lineIterator(file.getPath());
				} catch (IOException e) {
					throw new RuntimeException(e);

				}
			}
		});
		return allLinesIterator;
	}

}
