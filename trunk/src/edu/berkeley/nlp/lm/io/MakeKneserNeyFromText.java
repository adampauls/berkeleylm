package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.util.Logger;

public class MakeKneserNeyFromText
{

	/**
	 * 
	 */
	private static void usage() {
		System.err.println("Usage: <lmOrder> <ARPA lm output file> <textfiles>+");
		System.exit(1);
	}

	public static void main(final String[] argv) {
		if (argv.length < 3) {
			usage();
		}
		int lmOrder = Integer.parseInt(argv[0]);
		String outputFile = argv[1];
		List<File> inputFiles = new ArrayList<File>();
		for (int i = 2; i < argv.length; ++i) {
			inputFiles.add(new File(argv[i]));
		}
		Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
		Logger.startTrack("Reading text files " + inputFiles + " and writing to file " + outputFile);
		LmReaders.createKneserNeyLmFromTextFiles(inputFiles, new StringWordIndexer(), lmOrder, new File(outputFile));
		Logger.endTrack();
	}

}
