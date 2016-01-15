package edu.berkeley.nlp.lm.cache;

import edu.berkeley.nlp.lm.AbstractContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.bits.BitUtils;
import edu.berkeley.nlp.lm.util.MurmurHash;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

/**
 * This class wraps {@link ContextEncodedNgramLanguageModel} with a cache.
 * 
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public class ContextEncodedCachingLmWrapper<T> extends AbstractContextEncodedNgramLanguageModel<T>
{

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private final ContextEncodedLmCache contextCache;

	private final ContextEncodedNgramLanguageModel<T> lm;

	private final int capacity;

	/**
	 * This type of caching is only threadsafe if you have one cache wrapper per
	 * thread.
	 * 
	 * @param <T>
	 * @param lm
	 * @return
	 */
	public static <T> ContextEncodedCachingLmWrapper<T> wrapWithCacheNotThreadSafe(final ContextEncodedNgramLanguageModel<T> lm) {
		return wrapWithCacheNotThreadSafe(lm, 18);
	}

	public static <T> ContextEncodedCachingLmWrapper<T> wrapWithCacheNotThreadSafe(final ContextEncodedNgramLanguageModel<T> lm, final int cacheBits) {
		return new ContextEncodedCachingLmWrapper<T>(lm, false, cacheBits);
	}

	/**
	 * This type of caching is threadsafe and (internally) maintains a separate
	 * cache for each thread that calls it. Note each thread has its own cache,
	 * so if you have lots of threads, memory usage could be substantial.
	 * 
	 * @param <T>
	 * @param lm
	 * @return
	 */
	public static <T> ContextEncodedCachingLmWrapper<T> wrapWithCacheThreadSafe(final ContextEncodedNgramLanguageModel<T> lm) {
		return wrapWithCacheThreadSafe(lm, 16);
	}

	public static <T> ContextEncodedCachingLmWrapper<T> wrapWithCacheThreadSafe(final ContextEncodedNgramLanguageModel<T> lm, final int cacheBits) {
		return new ContextEncodedCachingLmWrapper<T>(lm, true, cacheBits);
	}

	private ContextEncodedCachingLmWrapper(final ContextEncodedNgramLanguageModel<T> lm, final boolean threadSafe, final int cacheBits) {
		this(lm, new ContextEncodedDirectMappedLmCache(cacheBits, threadSafe));
	}

	private ContextEncodedCachingLmWrapper(final ContextEncodedNgramLanguageModel<T> lm, final ContextEncodedLmCache cache) {
		super(lm.getLmOrder(), lm.getWordIndexer(), Float.NaN);
		this.lm = lm;
		this.contextCache = cache;
		capacity = contextCache.capacity();

	}

	@Override
	public WordIndexer<T> getWordIndexer() {
		return lm.getWordIndexer();
	}

	@Override
	public LmContextInfo getOffsetForNgram(final int[] ngram, final int startPos, final int endPos) {
		return lm.getOffsetForNgram(ngram, startPos, endPos);
	}

	@Override
	public int[] getNgramForOffset(final long contextOffset, final int contextOrder, final int word) {
		return lm.getNgramForOffset(contextOffset, contextOrder, word);
	}

	@Override
	public float getLogProb(final long contextOffset, final int contextOrder, final int word, @OutputParameter final LmContextInfo contextOutput) {
		if (contextOrder < 0) return lm.getLogProb(contextOffset, contextOrder, word, contextOutput);
		final int hash = hash(contextOffset, contextOrder, word) % capacity;
		float f = contextCache.getCached(contextOffset, contextOrder, word, hash, contextOutput);
		if (!Float.isNaN(f)) return f;
		f = lm.getLogProb(contextOffset, contextOrder, word, contextOutput);
		contextCache.putCached(contextOffset, contextOrder, word, f, hash, contextOutput);
		return f;
	}

	private static int hash(final long contextOffset, final int contextOrder, final int word) {
		final int hash = (int) MurmurHash.hashThreeLongs(contextOffset, contextOrder, word);
		return BitUtils.abs(hash);
	}

}
