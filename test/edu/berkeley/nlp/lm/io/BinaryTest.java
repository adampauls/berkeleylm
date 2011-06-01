package edu.berkeley.nlp.lm.io;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.util.Logger;

public class BinaryTest
{
	@Test
	public void testContextEncodedBinary() {
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		final ContextEncodedProbBackoffLm<String> lm = LmReaders.readContextEncodedLmFromArpa(FileUtils.getFile(PerplexityTest.BIG_TEST_ARPA).getPath(),
			new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
		File tmpFile = null;
		try {
			tmpFile = File.createTempFile("berkeleylmtest", "binary");
		} catch (IOException e) {
			Assert.fail(e.toString());

		}
		if (tmpFile != null) {
			tmpFile.deleteOnExit();
			IOUtils.writeObjFileHard(tmpFile, lm);
			@SuppressWarnings("unchecked")
			ContextEncodedProbBackoffLm<String> readLm = (ContextEncodedProbBackoffLm<String>) IOUtils.readObjFileHard(tmpFile);
			PerplexityTest.testContextEncodedLogProb(readLm, FileUtils.getFile(PerplexityTest.TEST_PERPLEX_TXT), PerplexityTest.TEST_PERPLEX_GOLD_PROB);
			tmpFile.delete();
		} else {
			Assert.fail();
		}
	}

	@Test
	public void testArrayEncodedBinary() {
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		for (boolean compress : new boolean[] { true, false }) {
			final ProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(FileUtils.getFile(PerplexityTest.BIG_TEST_ARPA).getPath(), compress,
				new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
			File tmpFile = null;
			try {
				tmpFile = File.createTempFile("berkeleylmtest", "binary");
			} catch (IOException e) {
				Assert.fail(e.toString());

			}
			if (tmpFile != null) {
				tmpFile.deleteOnExit();
				IOUtils.writeObjFileHard(tmpFile, lm);
				@SuppressWarnings("unchecked")
				ProbBackoffLm<String> readLm = (ProbBackoffLm<String>) IOUtils.readObjFileHard(tmpFile);
				PerplexityTest.testArrayEncodedLogProb(readLm, FileUtils.getFile(PerplexityTest.TEST_PERPLEX_TXT), PerplexityTest.TEST_PERPLEX_GOLD_PROB);
				tmpFile.delete();
			} else {
				Assert.fail();
			}
		}
	}
}
