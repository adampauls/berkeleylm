package edu.berkeley.nlp.lm.cache;

import edu.berkeley.nlp.lm.AbstractArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ArrayEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.util.MurmurHash;

/**
 * This class wraps {@link ArrayEncodedNgramLanguageModel} with a cache.
 * 
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class ArrayEncodedCachingLmWrapper<W> extends AbstractArrayEncodedNgramLanguageModel<W>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final ArrayEncodedLmCache cache;

	private final ArrayEncodedNgramLanguageModel<W> lm;

	/**
	 * To use this wrapper in a multithreaded environment, you should create one
	 * wrapper per thread.
	 * 
	 * @param <T>
	 * @param lm
	 * @return
	 */
	public static <W> ArrayEncodedCachingLmWrapper<W> wrapWithCacheNotThreadSafe(final ArrayEncodedNgramLanguageModel<W> lm) {
		return wrapWithCacheNotThreadSafe(lm, 18);
	}

	public static <W> ArrayEncodedCachingLmWrapper<W> wrapWithCacheNotThreadSafe(final ArrayEncodedNgramLanguageModel<W> lm, final int cacheBits) {
		return new ArrayEncodedCachingLmWrapper<W>(lm, false, cacheBits);
	}

	/**
	 * 
	 * This type of caching is threadsafe and (internally) maintains a separate
	 * cache for each thread that calls it. Note each thread has its own cache,
	 * so if you have lots of threads, memory usage could be substantial.
	 * 
	 * @param <W>
	 * @param lm
	 * @return
	 */
	public static <W> ArrayEncodedCachingLmWrapper<W> wrapWithCacheThreadSafe(final ArrayEncodedNgramLanguageModel<W> lm) {
		return wrapWithCacheThreadSafe(lm, 16);
	}

	public static <W> ArrayEncodedCachingLmWrapper<W> wrapWithCacheThreadSafe(final ArrayEncodedNgramLanguageModel<W> lm, final int cacheBits) {
		return new ArrayEncodedCachingLmWrapper<W>(lm, true, cacheBits);
	}

	private ArrayEncodedCachingLmWrapper(final ArrayEncodedNgramLanguageModel<W> lm, final boolean threadSafe, int cacheBits) {
		this(lm, new ArrayEncodedDirectMappedLmCache(cacheBits, lm.getLmOrder(), threadSafe));
	}

	private ArrayEncodedCachingLmWrapper(final ArrayEncodedNgramLanguageModel<W> lm, final ArrayEncodedLmCache cache) {
		super(lm.getLmOrder(), lm.getWordIndexer(), Float.NaN);
		this.cache = cache;
		this.lm = lm;

	}

	@Override
	public float getLogProb(final int[] ngram, final int startPos, final int endPos) {
		if (endPos - startPos == 1) return lm.getLogProb(ngram, startPos, endPos);
		final int hash = Math.abs(hash(ngram, startPos, endPos)) % cache.capacity();
		float f = cache.getCached(ngram, startPos, endPos, hash);
		if (!Float.isNaN(f)) return f;
		f = lm.getLogProb(ngram, startPos, endPos);
		cache.putCached(ngram, startPos, endPos, f, hash);
		return f;
	}

	private static int hash(final int[] key, final int startPos, final int endPos) {
		return MurmurHash.hash32(key, startPos, endPos);
	}

}
