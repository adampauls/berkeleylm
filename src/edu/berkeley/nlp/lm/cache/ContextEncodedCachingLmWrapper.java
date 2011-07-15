package edu.berkeley.nlp.lm.cache;

import edu.berkeley.nlp.lm.AbstractContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel;
import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

/**
 * This class wraps {@link ContextEncodedNgramLanguageModel} with a cache.
 * <p>
 * This wrapper is <b>not</b> threadsafe. To use a cache in a multithreaded
 * environment, you should create one wrapper per thread.
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

	/**
	 * To use a cache in a multithreaded environment, you should create one
	 * wrapper per thread.
	 * 
	 * @param <T>
	 * @param lm
	 * @return
	 */
	public static <T> ContextEncodedCachingLmWrapper<T> wrapWithCacheNotThreadSafe(final ContextEncodedNgramLanguageModel<T> lm) {
		return new ContextEncodedCachingLmWrapper<T>(lm);
	}

	/**
	 * To use a cache in a multithreaded environment, you should create one
	 * wrapper per thread.
	 * 
	 * @param <T>
	 * @param lm
	 * @return
	 */
	public static <T> ContextEncodedCachingLmWrapper<T> wrapWithCacheNotThreadSafe(final ContextEncodedNgramLanguageModel<T> lm,
		final ContextEncodedLmCache cache) {
		return new ContextEncodedCachingLmWrapper<T>(lm, cache);
	}

	private ContextEncodedCachingLmWrapper(final ContextEncodedNgramLanguageModel<T> lm) {
		this(lm, new ContextEncodedDirectMappedLmCache(24));
	}

	private ContextEncodedCachingLmWrapper(final ContextEncodedNgramLanguageModel<T> lm, final ContextEncodedLmCache cache) {
		super(lm.getLmOrder(), lm.getWordIndexer(), Float.NaN);
		this.lm = lm;
		this.contextCache = cache;

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
		final int hash = Math.abs(hash(contextOffset, contextOrder, word)) % contextCache.capacity();
		float f = contextCache.getCached(contextOffset, contextOrder, word, hash, contextOutput);
		if (!Float.isNaN(f)) return f;
		f = lm.getLogProb(contextOffset, contextOrder, word, contextOutput);
		contextCache.putCached(contextOffset, contextOrder, word, f, hash, contextOutput);
		return f;
	}

	private static int hash(final long contextOffset, final int contextOrder, final int word) {
		long hashCode = 1;
		hashCode = 13 * hashCode + word;
		hashCode = 13 * hashCode + contextOrder;
		hashCode = 13 * hashCode + contextOffset;
		return (int) hashCode;
	}

}
