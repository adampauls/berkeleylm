package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.ArrayEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;

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
		} catch (final IOException e) {
			Assert.fail(e.toString());

		}
		if (tmpFile != null) {
			tmpFile.deleteOnExit();
			IOUtils.writeObjFileHard(tmpFile, lm);
			@SuppressWarnings("unchecked")
			final ContextEncodedProbBackoffLm<String> readLm = (ContextEncodedProbBackoffLm<String>) IOUtils.readObjFileHard(tmpFile);
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
		for (final boolean compress : new boolean[] { true, false }) {
			final ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(FileUtils.getFile(PerplexityTest.BIG_TEST_ARPA).getPath(),
				compress, new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
			File tmpFile = null;
			try {
				tmpFile = File.createTempFile("berkeleylmtest", "binary");
			} catch (final IOException e) {
				Assert.fail(e.toString());

			}
			if (tmpFile != null) {
				tmpFile.deleteOnExit();
				IOUtils.writeObjFileHard(tmpFile, lm);
				@SuppressWarnings("unchecked")
				final ArrayEncodedProbBackoffLm<String> readLm = (ArrayEncodedProbBackoffLm<String>) IOUtils.readObjFileHard(tmpFile);
				PerplexityTest.testArrayEncodedLogProb(readLm, FileUtils.getFile(PerplexityTest.TEST_PERPLEX_TXT), PerplexityTest.TEST_PERPLEX_GOLD_PROB);
				tmpFile.delete();
			} else {
				Assert.fail();
			}
		}
	}
}
