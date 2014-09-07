package edu.berkeley.nlp.lm.io;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.zip.GZIPInputStream;

import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Computes the log probability of a list of files. With the <code>-g</code>
 * option, it interprets the next two arguments as a <code>vocab_cs.gz</code>
 * file (see {@link LmReaders} for more detail) and a Berkeley LM binary,
 * respectively. Without <code>-g</code>, it interprets the next file as a
 * Berkeley LM binary. All remaining files are treated as plain-text (possibly
 * gzipped) files which have one sentence per line; a dash is used to indicate
 * that text should from standard input. If no files are given, reads from
 * standard input.
 * 
 * @author adampauls
 * 
 */
public class ComputeLogProbabilityOfTextStream
{

	/**
	 * 
	 */
	private static void usage() {
		System.err.println("Usage: <Berkeley LM binary file> <outputfile>*\nor\n-g <vocab_cs file> <Google LM Binary>");
		System.exit(1);
	}

	public static void main(final String[] argv) throws FileNotFoundException, IOException {
		int i = 0;
		if (i >= argv.length) usage();
		boolean isGoogleBinary = false;
		if (argv[i].equals("-g")) {
			isGoogleBinary = true;
			i++;
		}
		if (i >= argv.length) usage();
		String vocabFile = null;
		if (isGoogleBinary) {
			vocabFile = argv[i++];
		}
		if (i >= argv.length) usage();
		String binaryFile = argv[i++];
		List<String> files = Arrays.asList(Arrays.copyOfRange(argv, i, argv.length));
		if (files.isEmpty()) files = Collections.singletonList("-");
		Logger.setGlobalLogger(new Logger.SystemLogger(System.err, System.err));
		NgramLanguageModel<String> lm = readBinary(isGoogleBinary, vocabFile, binaryFile);
		double prob = computeProb(files, lm);
		System.err.print("Log probability of text is: ");
		System.out.println(prob);
	}

	/**
	 * @param files
	 * @param lm
	 * @throws IOException
	 * @throws FileNotFoundException
	 */
	private static double computeProb(List<String> files, NgramLanguageModel<String> lm) throws IOException, FileNotFoundException {
		double logProb = 0.0;
		for (String file : files) {
			Logger.startTrack("Scoring file " + file + "; current log probability is " + logProb);
			final InputStream is = (file.equals("-")) ? System.in : (file.endsWith(".gz") ? new GZIPInputStream(new FileInputStream(file))
				: new FileInputStream(file));
			BufferedReader reader = new BufferedReader(new InputStreamReader(new BufferedInputStream(is)));
			for (String line : Iterators.able(IOUtils.lineIterator(reader))) {
				List<String> words = Arrays.asList(line.trim().split("\\s+"));
				logProb += lm.scoreSentence(words);
			}
			Logger.endTrack();
		}
		return logProb;
	}

	/**
	 * @param isGoogleBinary
	 * @param vocabFile
	 * @param binaryFile
	 * @return
	 */
	private static NgramLanguageModel<String> readBinary(boolean isGoogleBinary, String vocabFile, String binaryFile) {
		NgramLanguageModel<String> lm = null;
		if (isGoogleBinary) {
			Logger.startTrack("Reading Google Binary " + binaryFile + " with vocab " + vocabFile);
			lm = LmReaders.readGoogleLmBinary(binaryFile, vocabFile);
			Logger.endTrack();
		} else {
			Logger.startTrack("Reading LM Binary " + binaryFile);
			lm = LmReaders.readLmBinary(binaryFile);
			Logger.endTrack();
		}
		return lm;
	}

}
