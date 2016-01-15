package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.map.NgramMap.Entry;
import edu.berkeley.nlp.lm.util.Logger;
import edu.berkeley.nlp.lm.util.LongRef;
import edu.berkeley.nlp.lm.util.StrUtils;
import edu.berkeley.nlp.lm.values.KneserNeyCountValueContainer;
import edu.berkeley.nlp.lm.values.KneserNeyCountValueContainer.KneserNeyCounts;
import edu.berkeley.nlp.lm.values.ProbBackoffPair;

/**
 * Class for producing a Kneser-Ney language model in ARPA format from raw text.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class KneserNeyFileWritingLmReaderCallback<W> implements ArpaLmReaderCallback<ProbBackoffPair>
{

	private PrintWriter out;

	private WordIndexer<W> wordIndexer;

	public KneserNeyFileWritingLmReaderCallback(final File outputFile, final WordIndexer<W> wordIndexer) {
		this(IOUtils.openOutHard(outputFile), wordIndexer);
	}

	public KneserNeyFileWritingLmReaderCallback(final PrintWriter out, final WordIndexer<W> wordIndexer) {
		this.wordIndexer = wordIndexer;
		this.out = out;

	}

	@Override
	public void handleNgramOrderFinished(int order) {
		out.println("");
	}

	@Override
	public void handleNgramOrderStarted(int order) {
		out.println("\\" + (order) + "-grams:");
	}

	@Override
	public void call(int[] ngram, int startPos, int endPos, ProbBackoffPair value, String words) {
		final String line = StrUtils.join(WordIndexer.StaticMethods.toList(wordIndexer, ngram, startPos, endPos));
		final boolean endsWithEndSym = ngram[ngram.length - 1] == wordIndexer.getIndexPossiblyUnk(wordIndexer.getEndSymbol());
		if (endsWithEndSym || value.backoff == 0.0f)
			out.printf(Locale.US, "%f\t%s\n", value.prob, line);
		else {
			out.printf(Locale.US, "%f\t%s\t%f\n", value.prob, line, value.backoff);
		}
	}

	@Override
	public void cleanup() {
		out.println("\\end\\");
		out.close();
	}

	@Override
	public void initWithLengths(List<Long> numNGrams) {
		Logger.startTrack("Writing ARPA");
		out.println();
		out.println("\\data\\");
		for (int ngramOrder = 0; ngramOrder < numNGrams.size(); ++ngramOrder) {
			final long numNgrams = numNGrams.get(ngramOrder);
			out.println("ngram " + (ngramOrder + 1) + "=" + numNgrams);
		}
		out.println();
	}

}
