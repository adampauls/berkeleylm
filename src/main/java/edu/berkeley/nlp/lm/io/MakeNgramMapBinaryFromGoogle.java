package edu.berkeley.nlp.lm.io;

import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Like {@link MakeLmBinaryFromGoogle}, except it only writes the NgramMap
 * portion of the LM, meaning the binary does not contain the vocabulary. We
 * have used this internally to build binaries that we provide for download.
 * Since these binaries are useless without the vocabulary provided with the
 * Google n-gram corpus, we can distribute them without incurring the wrath of
 * the LDC.
 * <p>
 * These binaries can be read in used
 * {@link LmReaders#readGoogleLmBinary(String, edu.berkeley.nlp.lm.WordIndexer, String)}
 * 
 * @author adampauls
 * 
 */
public class MakeNgramMapBinaryFromGoogle
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
		final StupidBackoffLm<String> lm = (StupidBackoffLm<String>) LmReaders.readLmFromGoogleNgramDir(lmFile, true, false);
		Logger.endTrack();
		final String outFile = argv[1];
		Logger.startTrack("Writing to file " + outFile + " . . . ");
		IOUtils.writeObjFileHard(outFile, lm.getNgramMap());
		Logger.endTrack();

	}
}
