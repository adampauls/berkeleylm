package edu.berkeley.nlp.lm.io;

import org.junit.Test;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.StringWordIndexer;

public class TestMain
{

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		final String googleDir = "/Users/adampauls/Downloads/Ngram dataset";
		final StringWordIndexer wordIndexer = new StringWordIndexer();
		LmReaders.readLmFromGoogleNgramDir(googleDir, false, true, wordIndexer, new ConfigOptions());
	}

}
