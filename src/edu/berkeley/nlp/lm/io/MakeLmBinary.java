package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;

import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.util.Logger;

public class MakeLmBinary
{
	public static void main(final String[] argv) throws IOException {

		if (argv.length != 2) {
			System.err.println("Expecting 2 args: <ARPA lm file> <outputfile>");
			System.exit(1);
		}
		Logger.setGlobalLogger(new Logger.SystemLogger(System.out, System.err));
		Logger.startTrack("Reading Lm File " + argv[0] + " . . . ");
		final ContextEncodedProbBackoffLm<String> lm = LmReaders.readContextEncodedLmFromArpa(argv[0], false);
		Logger.endTrack();
		Logger.startTrack("Writing to file " + argv[1] + " . . . ");
		IOUtils.writeObjFile(new File(argv[1]), lm);
		Logger.endTrack();

		//read with IOUtils.readObjFile(new File(argv[1]))
	}

}
