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

public class PerplexityTest
{
	public static final String TEST_PERPLEX_TINY_TXT = "test_perplex_tiny.txt";

	public static final String TEST_PERPLEX_TXT = "test_perplex.txt";

	public static final String BIG_TEST_ARPA = "big_test.arpa";
	

	public static final float TEST_PERPLEX_GOLD_PROB = -2675.41f;

	public static final float TEST_PERPLEX_TINY_GOLD_PROB = -38.9312f;

	@Test
	public void testTiny() {
		File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		ArrayEncodedProbBackoffLm<String> lm = getLm();
		testArrayEncodedLogProb(lm, file, goldLogProb);
		//		Assert.assertEquals(logScore, -2806.4f, 1e-1);
	}

	@Test
	public void testTinyContextEncoded() {
		File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm();
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void test() {
		File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		ArrayEncodedProbBackoffLm<String> lm = getLm();
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCompressed() {
		File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), true, new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCompressedCached() {
		File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), true, new StringWordIndexer(), configOptions, Integer.MAX_VALUE);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm, new ArrayEncodedDirectMappedLmCache(16, lm.getLmOrder())), file,
			goldLogProb);
	}

	@Test
	public void testContextEncoded() {
		File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm();
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCachedTiny() {
		File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm_ = getLm();
		ArrayEncodedNgramLanguageModel<String> lm = ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_,
			new ArrayEncodedDirectMappedLmCache(16, lm_.getLmOrder()));
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCachedTinyContextEncoded() {
		File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		ContextEncodedCachingLmWrapper<String> lm = ContextEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(getContextEncodedLm(),
			new ContextEncodedDirectMappedLmCache(16));
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCached() {
		File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm_ = getLm();
		ArrayEncodedNgramLanguageModel<String> lm = ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(getLm(), new ArrayEncodedDirectMappedLmCache(16,
			lm_.getLmOrder()));
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCachedContextEncoded() {
		File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		ContextEncodedCachingLmWrapper<String> lm = ContextEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(getContextEncodedLm(),
			new ContextEncodedDirectMappedLmCache(16));
		testContextEncodedLogProb(lm, file, goldLogProb);
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
	public static void testContextEncodedLogProb(ContextEncodedNgramLanguageModel<String> lm_, File file, float goldLogProb) {
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
					final float score2 = lm_.getLogProb(context.offset, context.order, sent[i], null);
					final float score = lm_.getLogProb(context.offset, context.order, sent[i], context);
					Assert.assertEquals(score, score2, Float.MIN_VALUE);
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

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	public static void testArrayEncodedLogProb(ArrayEncodedNgramLanguageModel<String> lm_, File file, float goldLogProb) {
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
