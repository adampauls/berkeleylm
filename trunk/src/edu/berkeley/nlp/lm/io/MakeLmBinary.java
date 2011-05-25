package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;

import edu.berkeley.nlp.lm.BackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.map.ConfigOptions;

public class MakeLmBinary
{
	public static void main(String[] argv) throws IOException {
		if (argv.length != 2) {
			System.err.println("Expecting 2 args: <ARPA lm file> <outputfile>");
			System.exit(1);
		}
		System.out.print("Reading Lm File " + argv[0] + " . . . ");
		BackoffLm<String> lm = LmReaders.readArpaLmFile(argv[0]);
		System.out.println("done.");
		System.out.print("Writing to file " + argv[1] + " . . . ");
		IOUtils.writeObjFile(new File(argv[1]), lm);
		System.out.println("done.");
	}

}
