package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ArrayEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.cache.ArrayEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.cache.ArrayEncodedDirectMappedLmCache;
import edu.berkeley.nlp.lm.cache.ContextEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.cache.ContextEncodedDirectMappedLmCache;
import edu.berkeley.nlp.lm.collections.Iterators;

public class MissingEntryTest
{

	public static final String BIG_TEST_ARPA = "missing_test_fourgram.arpa";

	@Test
	public void testArrayEncoded() {
		File file = FileUtils.getFile(BIG_TEST_ARPA);

		ArrayEncodedProbBackoffLm<String> lm = getLm();
		testArrayEncodedLogProb(lm, file);
		//		Assert.assertEquals(logScore, -2806.4f, 1e-1);
	}

	@Test
	public void testContextEncoded() {
		File file = FileUtils.getFile(BIG_TEST_ARPA);

		ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm();
		testContextEncodedLogProb(lm, file);
		//		Assert.assertEquals(logScore, -2806.4f, 1e-1);
	}

	/**
	 * @return
	 */
	private ContextEncodedProbBackoffLm<String> getContextEncodedLm() {
		File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		ContextEncodedProbBackoffLm<String> lm = LmReaders.readContextEncodedLmFromArpa(lmFile.getPath(), new StringWordIndexer(), configOptions,
			Integer.MAX_VALUE);
		return lm;
	}

	/**
	 * @return
	 */
	private ArrayEncodedProbBackoffLm<String> getLm() {
		File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), false, new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
		return lm;
	}

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	public static void testArrayEncodedLogProb(ArrayEncodedNgramLanguageModel<String> lm_, File file) {

		Assert.assertEquals(lm_.getLogProb(Arrays.asList("This another test is".split(" "))), -0.67443009, 1e-2);
		Assert.assertEquals(lm_.getLogProb(Arrays.asList("another test sentence.".split(" "))), -0.07443009, 1e-2);
		Assert.assertEquals(lm_.getLogProb(Arrays.asList("is another test".split(" "))), -0.1366771, 1e-2);
		Assert.assertEquals(lm_.getLogProb(Arrays.asList("another test".split(" "))), -0.60206 + -0.2218488, 1e-2);
	}

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	public static void testContextEncodedLogProb(ContextEncodedNgramLanguageModel<String> lm_, File file) {

		Assert.assertEquals(lm_.getLogProb(Arrays.asList("This another test is".split(" "))), -0.67443009, 1e-2);
		Assert.assertEquals(lm_.getLogProb(Arrays.asList("another test sentence.".split(" "))), -0.07443009, 1e-2);
		Assert.assertEquals(lm_.getLogProb(Arrays.asList("is another test".split(" "))), -0.1366771, 1e-2);
		Assert.assertEquals(lm_.getLogProb(Arrays.asList("another test".split(" "))), -0.60206 + -0.2218488, 1e-2);
	}
}
