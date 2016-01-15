package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.ArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ArrayEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.NgramLanguageModel;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo;
import edu.berkeley.nlp.lm.ContextEncodedProbBackoffLm;
import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.cache.ArrayEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.cache.ArrayEncodedDirectMappedLmCache;
import edu.berkeley.nlp.lm.cache.ContextEncodedCachingLmWrapper;
import edu.berkeley.nlp.lm.cache.ContextEncodedDirectMappedLmCache;
import edu.berkeley.nlp.lm.collections.Counter;
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
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm = getLm(false);
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testTinyUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm = getLm(true);
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testBigNgram() {
		final ArrayEncodedProbBackoffLm<String> lm = getLm(true);
		final List<String> ngram = Arrays.asList("they they are wasting our money".split(" "));
		final int[] longNgram = NgramLanguageModel.StaticMethods.toIntArray(ngram, lm);
		float logProb = lm.getLogProb(longNgram);
		Assert.assertEquals(-0.07232222, logProb,1e-7);
	}

	@Test
	public void testTinyContextEncoded() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm(false);
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testTinyContextEncodedUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm(true);
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void test() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm = getLm(false);
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}
	
	@Test
	public void testPredict() {
		final ArrayEncodedProbBackoffLm<String> lm = getLm(false);
		Counter<String> c = NgramLanguageModel.StaticMethods.getDistributionOverNextWords(lm, Arrays.asList("this is some context and the release of".split(" ")));
		Assert.assertEquals(c.getCount("political"), 0.8381973309763864, 1e-10);
	}

	@Test
	public void testUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm = getLm(true);
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCompressed() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		final ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), true, new StringWordIndexer(), configOptions,
			Integer.MAX_VALUE);
		testArrayEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCompressedCached() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.unknownWordLogProb = 0.0f;
		final ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), true, new StringWordIndexer(), configOptions,
			Integer.MAX_VALUE);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm, 16), file, goldLogProb);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm, 16), file, goldLogProb);
	}

	@Test
	public void testContextEncoded() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm(false);
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testContextEncodedUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm = getContextEncodedLm(true);
		testContextEncodedLogProb(lm, file, goldLogProb);
	}

	@Test
	public void testCachedTiny() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm_ = getLm(false);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCachedTinyUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm_ = getLm(true);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCachedTinyContextEncoded() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm_ = getContextEncodedLm(false);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCachedTinyContextEncodedUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TINY_TXT);
		final float goldLogProb = TEST_PERPLEX_TINY_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm_ = getContextEncodedLm(true);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCached() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm_ = getLm(false);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCachedUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ArrayEncodedProbBackoffLm<String> lm_ = getLm(true);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testArrayEncodedLogProb(ArrayEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCachedContextEncoded() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm_ = getContextEncodedLm(false);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	@Test
	public void testCachedContextEncodedUnranked() {
		final File file = FileUtils.getFile(TEST_PERPLEX_TXT);
		final float goldLogProb = TEST_PERPLEX_GOLD_PROB;
		final ContextEncodedProbBackoffLm<String> lm_ = getContextEncodedLm(true);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheNotThreadSafe(lm_, 16), file, goldLogProb);
		testContextEncodedLogProb(ContextEncodedCachingLmWrapper.wrapWithCacheThreadSafe(lm_, 16), file, goldLogProb);
	}

	/**
	 * @return
	 */
	private ContextEncodedProbBackoffLm<String> getContextEncodedLm(boolean unranked) {
		final File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.storeRankedProbBackoffs = !unranked;
		configOptions.unknownWordLogProb = 0.0f;
		final ContextEncodedProbBackoffLm<String> lm = LmReaders.readContextEncodedLmFromArpa(lmFile.getPath(), new StringWordIndexer(), configOptions,
			Integer.MAX_VALUE);
		return lm;
	}

	/**
	 * @return
	 */
	private ArrayEncodedProbBackoffLm<String> getLm(boolean unranked) {
		final File lmFile = FileUtils.getFile(BIG_TEST_ARPA);
		final ConfigOptions configOptions = new ConfigOptions();
		configOptions.storeRankedProbBackoffs = !unranked;
		configOptions.unknownWordLogProb = 0.0f;
		final ArrayEncodedProbBackoffLm<String> lm = LmReaders.readArrayEncodedLmFromArpa(lmFile.getPath(), false, new StringWordIndexer(), configOptions,
			Integer.MAX_VALUE);
		return lm;
	}

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	public static void testContextEncodedLogProb(final ContextEncodedNgramLanguageModel<String> lm_, final File file, final float goldLogProb) {
		float logScore = 0.0f;
		try {
			for (final String line : Iterators.able(IOUtils.lineIterator(file.getPath()))) {

				final String[] split = line.trim().split(" ");
				final int[] sent = new int[split.length + 2];
				sent[0] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getStartSymbol());
				sent[sent.length - 1] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getEndSymbol());
				int k = 1;
				for (final String s : split) {
					sent[k++] = lm_.getWordIndexer().getIndexPossiblyUnk(s);

				}
				final LmContextInfo context = new LmContextInfo();
				lm_.getLogProb(context.offset, context.order, sent[0], context);
				float sentScore = 0.0f;
				for (int i = 1; i < sent.length; ++i) {
					final float score2 = lm_.getLogProb(context.offset, context.order, sent[i], null);
					final float score = lm_.getLogProb(context.offset, context.order, sent[i], context);
					Assert.assertEquals(score, score2, Float.MIN_VALUE);
					sentScore += score;
				}
				Assert.assertEquals(sentScore, lm_.scoreSentence(Arrays.asList(split)), 1e-5);
				logScore += sentScore;

			}
		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
		Assert.assertEquals(logScore, goldLogProb, 1e-1);
	}

	/**
	 * @param lm_
	 * @param file
	 * @param goldLogProb
	 */
	public static void testArrayEncodedLogProb(final ArrayEncodedNgramLanguageModel<String> lm_, final File file, final float goldLogProb) {
		float logScore = 0.0f;
		try {
			for (final String line : Iterators.able(IOUtils.lineIterator(file.getPath()))) {
				final String[] split = line.trim().split(" ");
				final int[] sent = new int[split.length + 2];
				sent[0] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getStartSymbol());
				sent[sent.length - 1] = lm_.getWordIndexer().getOrAddIndexFromString(lm_.getWordIndexer().getEndSymbol());
				int k = 1;
				for (final String s : split) {
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
				Assert.assertEquals(sentScore, lm_.scoreSentence(Arrays.asList(split)), 1e-5);
				logScore += sentScore;

			}
		} catch (final IOException e) {
			throw new RuntimeException(e);

		}
		Assert.assertEquals(logScore, goldLogProb, 1e-1);
	}
}
