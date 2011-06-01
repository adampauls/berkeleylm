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

public class MakeLmBinaryFromArpa
{
	private static final String CONTEXT_OPT = "-e";

	private static final String COMPRESS_OPT = "-c";

	/**
	 * 
	 */
	private static void usage() {
		System.err.println("Usage: [opts] <ARPA lm file> <outputfile>");
		System.err.println("\t-c: build a compressed LM instead of the default hash table ");
		System.err.println("\t-e: build a context-encoded LM instead of the array encoding");
		System.exit(1);
	}

	public static void main(final String[] argv) {
		List<String> fileArgs = new ArrayList<String>();
		Set<String> opts = new HashSet<String>();
		for (String arg : argv) {
			if (arg.startsWith("-"))
				opts.add(arg);
			else
				fileArgs.add(arg);
		}
		if (fileArgs.size() != 2) {
			usage();
		}
		opts.removeAll(Arrays.asList(COMPRESS_OPT, CONTEXT_OPT));
		if (!opts.isEmpty()) {
			System.err.println("Unrecognized opts: " + opts);
			usage();
		}
		boolean compress = (opts.contains(COMPRESS_OPT));
		boolean contextEncode = (opts.contains(CONTEXT_OPT));
		if (compress && contextEncode) {
			System.err.print("Context-encoded LM does not support compression.");
			System.exit(1);
		}
		Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
		Logger.startTrack("Reading Lm File " + argv[0] + " . . . ");
		final String lmFile = fileArgs.get(0);
		final NgramLanguageModel<String> lm = contextEncode ? LmReaders.readContextEncodedLmFromArpa(lmFile) : LmReaders.readArrayEncodedLmFromArpa(lmFile,
			compress);
		Logger.endTrack();
		final String outFile = fileArgs.get(1);
		Logger.startTrack("Writing to file " + outFile + " . . . ");
		LmReaders.writeLmBinary(lm, outFile);
		Logger.endTrack();

	}

}
