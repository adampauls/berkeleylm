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
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Given a directory in Google n-grams format, builds a binary representation of
 * a stupid-backoff language model language model and writes it to disk.
 * Language model binaries are significantly smaller and faster to load. Note:
 * actually running this code on the full Google-ngrams corpus can be very slow
 * and memory intensive -- on our machines, it takes about 32GB of memory and
 * 15 hours.
 * 
 * Note that if the input/output files have a .gz suffix, they will be
 * unzipped/zipped as necessary.
 * 
 * @author adampauls
 * 
 */
public class MakeLmBinaryFromGoogle
{

	/**
	 * 
	 */
	private static void usage() {
		System.err.println("Usage: <Google n-grams dir> <outputfile>");
		System.exit(1);
	}

	public static void main(final String[] argv) {
		if (argv.length != 2) usage();
		Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
		Logger.startTrack("Reading Lm File " + argv[0] + " . . . ");
		final String lmFile = argv[1];
		final NgramLanguageModel<String> lm = LmReaders.readLmFromGoogleNgramDir(lmFile, true);
		Logger.endTrack();
		final String outFile = argv[1];
		Logger.startTrack("Writing to file " + outFile + " . . . ");
		LmReaders.writeLmBinary(lm, outFile);
		Logger.endTrack();

	}
}
