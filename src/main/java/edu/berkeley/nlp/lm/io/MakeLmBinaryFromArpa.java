package edu.berkeley.nlp.lm.io;

import java.util.ArrayList;
import java.util.List;

import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Given a language model in ARPA format, builds a binary representation of the
 * language model and writes it to disk. Language model binaries are
 * significantly smaller and faster to load than ARPA files.
 * <p>
 * Note that if the input/output files have a <code>.gz</code> suffix, they will
 * be unzipped/zipped as necessary.
 * 
 * @author adampauls
 * 
 */
public class MakeLmBinaryFromArpa
{

	private enum Opts
	{
		HASH_OPT
		{
			@Override
			public String toString() {
				return "-h";
			}

			@Override
			public String docString() {
				return "build an array-encoded hash-table LM (the default)";
			}

			@Override
			public NgramLanguageModel<String> makeLm(final String file) {
				return LmReaders.readArrayEncodedLmFromArpa(file, false);
			}
		},
		CONTEXT_OPT
		{
			@Override
			public String toString() {
				return "-e";
			}

			@Override
			public String docString() {
				return "build a context-encoded LM instead of the default hash table";
			}

			@Override
			public NgramLanguageModel<String> makeLm(final String file) {
				return LmReaders.readContextEncodedLmFromArpa(file);
			}
		},
		COMPRESS_OPT
		{
			@Override
			public String toString() {
				return "-c";
			}

			@Override
			public String docString() {
				return "build a compressed hash-table LM instead of the array encoding";
			}

			@Override
			public NgramLanguageModel<String> makeLm(final String file) {
				return LmReaders.readArrayEncodedLmFromArpa(file, true);
			}
		};

		public abstract String docString();

		public abstract NgramLanguageModel<String> makeLm(String file);

	}

	/**
	 * 
	 */
	private static void usage() {
		System.err.println("Usage: [opts] <ARPA lm file> <outputfile>");
		for (final Opts opts : Opts.values()) {
			System.err.println("\t" + opts.toString() + ": " + opts.docString());
		}
		System.exit(1);
	}

	public static void main(final String[] argv) {
		final List<String> fileArgs = new ArrayList<String>();
		Opts finalOpt = Opts.HASH_OPT;
		OUTER: for (final String arg : argv) {
			if (arg.startsWith("-")) {
				for (final Opts opts : Opts.values()) {
					if (opts.toString().equals(arg)) {
						finalOpt = opts;
						continue OUTER;
					}
				}
				System.err.println("Unrecognized opts: " + arg);
				usage();
			} else
				fileArgs.add(arg);
		}
		if (fileArgs.size() != 2) {
			usage();
		}

		Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
		final String lmFile = fileArgs.get(0);
		Logger.startTrack("Reading Lm File " + lmFile + " . . . ");
		final NgramLanguageModel<String> lm = finalOpt.makeLm(lmFile);
		Logger.endTrack();
		final String outFile = fileArgs.get(1);
		Logger.startTrack("Writing to file " + outFile + " . . . ");
		LmReaders.writeLmBinary(lm, outFile);
		Logger.endTrack();

	}
}
