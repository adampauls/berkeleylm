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
import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.ProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.cache.CachingLmWrapper;
import edu.berkeley.nlp.lm.cache.ContextEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.collections.Iterators;

public class PerplexityTest
{
	final float bigGoldProb = -2675.41f;

	final float smallGoldProb = -38.9312f;

	@Test
	public void testTiny() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex_tiny.txt");
		float goldLogProb = smallGoldProb;
		ProbBackoffLm<String> lm = getLm();
		testLogProb(lm, file, goldLogProb);
		//		Assert.assertEquals(logScore, -2806.4f, 1e-1);
	}

	@Test
	public void testTinyContextEncoded() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex_tiny.txt");
		float goldLogProb = smallGoldProb;
		ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm();
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void test() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex.txt");
		float goldLogProb = bigGoldProb;
		ProbBackoffLm<String> lm = getLm();
		testLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testContextEncoded() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex.txt");
		float goldLogProb = bigGoldProb;
		ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm();
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCachedTiny() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex_tiny.txt");
		float goldLogProb = smallGoldProb;
		NgramLanguageModel<String> lm = new CachingLmWrapper<String>(getLm());
		testLogProb(lm, file, goldLogProb);
		//		Assert.assertEquals(logScore, -2806.4f, 1e-1);
	}

	@Test
	public void testCachedTinyContextEncoded() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex_tiny.txt");
		float goldLogProb = smallGoldProb;
		ContextEncodedCachingLmWrapper<String> lm = new ContextEncodedCachingLmWrapper<String>(getContextEncodedLm());
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCached() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex.txt");
		float goldLogProb = bigGoldProb;
		NgramLanguageModel<String> lm = new CachingLmWrapper<String>(getLm());
		testLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCachedContextEncoded() {
		File file = KneserNeyFromTextReaderTest.getFile("test_perplex.txt");
		float goldLogProb = bigGoldProb;
		ContextEncodedCachingLmWrapper<String> lm = new ContextEncodedCachingLmWrapper<String>(getContextEncodedLm());
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	/**
	 * @return
	 */
	private ContextEncodedProbBackoffLm<String> getContextEncodedLm() {
		File lmFile = KneserNeyFromTextReaderTest.getFile("big_test.arpa");
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		ContextEncodedProbBackoffLm<String> lm = LmReaders.readContextEncodedLmFromArpa(lmFile.getPath(), false, new StringWordIndexer(), configOptions,
			Integer.MAX_VALUE);
		return lm;
	}

	/**
	 * @return
	 */
	private ProbBackoffLm<String> getLm() {
		File lmFile = KneserNeyFromTextReaderTest.getFile("big_test.arpa");
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		ProbBackoffLm<String> lm = LmReaders.readLmFromArpa(lmFile.getPath(), false, new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
		return lm;
	}

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	private void testContextEncodedLogProb(ContextEncodedNgramLanguageModel<String> lm_, File file, float goldLogProb) {
		float logScore = 0.0f;
		try {
			for (final String line : Iterators.able(IOUtils.lineIterator(file.getPath()))) {

				final String[] split = line.trim().split(" ");
				int[] sent = new int[split.length + 2];
				sent[0] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getStartSymbol());
				sent[sent.length - 1] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getEndSymbol());
				int k = 1;
				for (String s : split) {
					sent[k++] = lm_.getWordIndexer().getIndexPossiblyUnk(s);

				}
				LmContextInfo context = new LmContextInfo();
				lm_.getLogProb(context.offset, context.order, sent[0], context);
				float sentScore = 0.0f;
				for (int i = 1; i < sent.length; ++i) {
					final float score = lm_.getLogProb(context.offset, context.order, sent[i], context);
					sentScore += score;
					System.out.println(score);
				}
				Assert.assertEquals(sentScore, lm_.scoreSentence(Arrays.asList(split)), 1e-4);
				logScore += sentScore;

			}
		} catch (IOException e) {
			throw new RuntimeException(e);

		}
		System.out.println("******");
		Assert.assertEquals(logScore, goldLogProb, 1e-1);
	}

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	private void testLogProb(NgramLanguageModel<String> lm_, File file, float goldLogProb) {
		float logScore = 0.0f;
		try {
			for (final String line : Iterators.able(IOUtils.lineIterator(file.getPath()))) {
				final String[] split = line.trim().split(" ");
				int[] sent = new int[split.length + 2];
				sent[0] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getStartSymbol());
				sent[sent.length - 1] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getEndSymbol());
				int k = 1;
				for (String s : split) {
					sent[k++] = lm_.getWordIndexer().getIndexPossiblyUnk(s);

				}
				float sentScore = 0.0f;
				for (int i = 2; i <= Math.min(lm_.getLmOrder(), sent.length); ++i) {
					final float score = lm_.getLogProb(sent, 0, i);
					sentScore += score;
				}
				for (int i = 1; i <= sent.length - lm_.getLmOrder(); ++i) {
					final float score = lm_.getLogProb(sent, i, i + lm_.getLmOrder());
					sentScore += score;
				}
				Assert.assertEquals(sentScore, lm_.scoreSentence(Arrays.asList(split)), 1e-4);
				logScore += sentScore;

			}
		} catch (IOException e) {
			throw new RuntimeException(e);

		}
		Assert.assertEquals(logScore, goldLogProb, 1e-1);
	}
}
