package edu.berkeley.nlp.lm.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

/**
 * A parser for ARPA LM files.
 * 
 * @author Alex Bouchard-Cote
 * @author Adam Pauls
 */
public class ArpaLmReader<W> implements LmReader<ProbBackoffPair, ArpaLmReaderCallback<ProbBackoffPair>>
{

	public static final String START_SYMBOL = "<s>";

	public static final String END_SYMBOL = "</s>";

	public static final String UNK_SYMBOL = "<unk>";

	private BufferedReader reader;

	private int currentNGramLength = 1;

	int currentNGramCount = 0;

	/**
	 * The current line in the file being examined.
	 */
	private int lineNumber = 1;

	private final WordIndexer<W> wordIndexer;

	private final int maxOrder;

	private final String file;

	/**
	 * 
	 * @return
	 * @throws IOException
	 */
	protected String readLine() throws IOException {
		lineNumber++;
		return reader.readLine();
	}

	/**
	 * 
	 * @param reader
	 */
	public ArpaLmReader(final String file, final WordIndexer<W> wordIndexer, final int maxNgramOrder) {
		this.file = file;
		this.wordIndexer = wordIndexer;
		this.maxOrder = maxNgramOrder;
	}

	/**
	 * Parse the ARPA file and populate the relevant fields of the enclosing
	 * ICSILanguageModel
	 * 
	 */
	@Override
	public void parse(final ArpaLmReaderCallback<ProbBackoffPair> callback) {
		currentNGramLength = 1;
		currentNGramCount = 0;
		lineNumber = 1;
		this.reader = IOUtils.openInHard(file);
		Logger.startTrack("Parsing ARPA language model file");
		final List<Long> numNGrams = parseHeader();
		callback.initWithLengths(numNGrams);
		parseNGrams(callback);
		Logger.endTrack();
		callback.cleanup();
		wordIndexer.setStartSymbol(wordIndexer.getWord(wordIndexer.getOrAddIndexFromString(START_SYMBOL)));
		wordIndexer.setEndSymbol(wordIndexer.getWord(wordIndexer.getOrAddIndexFromString(END_SYMBOL)));
		wordIndexer.setUnkSymbol(wordIndexer.getWord(wordIndexer.getOrAddIndexFromString(UNK_SYMBOL)));
	}

	/**
	 * 
	 * @param callback
	 * @throws IOException
	 * @throws ARPAParserException
	 */
	protected List<Long> parseHeader() {
		final List<Long> numEachNgrams = new ArrayList<Long>();
		try {
			String readLine = null;
			while ((readLine = readLine()) != null) {

				final String ngramTotalPrefix = "ngram ";
				if (readLine.startsWith(ngramTotalPrefix)) {
					final int equalsIndex = readLine.indexOf('=');
					assert equalsIndex >= 0;
					final long currNumNGrams = Long.parseLong(readLine.substring(equalsIndex + 1));
					if (numEachNgrams.size() < maxOrder) numEachNgrams.add(currNumNGrams);
				}
				if (readLine.contains("\\1-grams:")) { return numEachNgrams; }
			}
		} catch (final NumberFormatException e) {
			throw new RuntimeException(e);

		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
		throw new RuntimeException("Something wrong with I/O.");
	}

	/**
		 * 
		 * 
		 */
	protected void parseNGrams(final ArpaLmReaderCallback<ProbBackoffPair> callback) {

		int currLine = 0;
		Logger.startTrack("Reading 1-grams");
		callback.handleNgramOrderStarted(currentNGramLength);
		try {
			String line = null;
			int[] ngramScratch = new int[currentNGramLength];
			while ((line = reader.readLine()) != null) {
				if (currLine % 100000 == 0) Logger.logs("Read " + currLine + " lines");
				currLine++;
				if (line.length() == 0) {
					// nothing to do (skip blank lines)
				} else if (line.charAt(0) == '\\') {
					// a new block of n-gram is beginning
					if (!line.startsWith("\\end")) {
						Logger.logs(currentNGramCount + " " + currentNGramLength + "-gram read.");
						Logger.endTrack();
						callback.handleNgramOrderFinished(currentNGramLength);
						currentNGramLength++;
						if (currentNGramLength > maxOrder) return;
						ngramScratch = new int[currentNGramLength];
						currentNGramCount = 0;
						callback.handleNgramOrderStarted(currentNGramLength);
						Logger.startTrack("Reading " + currentNGramLength + "-grams");
					}
				} else {
					parseLine(callback, line, ngramScratch);
				}
			}
			reader.close();
		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
		Logger.endTrack();
		callback.handleNgramOrderFinished(currentNGramLength);
	}

	/**
	 * 
	 * @param line
	 * @throws ARPAParserException
	 */
	private void parseLine(final ArpaLmReaderCallback<ProbBackoffPair> callback, final String line, final int[] ngram) {
		// this is a 2 or 3 columns n-gram entry
		final int firstTab = line.indexOf('\t');
		final int secondTab = line.indexOf('\t', firstTab + 1);
		final boolean hasBackOff = (secondTab >= 0);

		final int length = line.length();
		parseNGram(line, firstTab + 1, secondTab < 0 ? length : secondTab, ngram);

		// the first column contains the log pr
		final String logProbString = line.substring(0, firstTab);
		final float logProbability = Float.parseFloat(logProbString);
		float backoff = 0.0f;

		// and its backoff, if specified
		if (hasBackOff) {
			backoff = Float.parseFloat(line.substring(secondTab + 1, length));
		}
		// add the new n-gram
		if (logProbability > 0.0) throw new RuntimeException("Bad ARPA line " + line);
		callback.call(ngram, 0, ngram.length, new ProbBackoffPair(logProbability, backoff), line);

		currentNGramCount++;
	}

	/**
	 * 
	 * @param string
	 * @return
	 */
	private void parseNGram(final String string, int start, int stringLength, final int[] retVal) {
		int k = 0;
		int spaceIndex = start;
		while (true) {
			final int nextIndex = string.indexOf(' ', spaceIndex);
			final String currWord = string.substring(spaceIndex, nextIndex < 0 ? stringLength : nextIndex);
			retVal[k++] = wordIndexer.getOrAddIndexFromString(currWord);
			if (nextIndex < 0) break;
			spaceIndex = nextIndex + 1;
		}
	}

}