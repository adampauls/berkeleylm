package edu.berkeley.nlp.lm.map;

import java.io.Serializable;
import java.util.Arrays;

import edu.berkeley.nlp.lm.ConfigOptions;
import edu.berkeley.nlp.lm.values.ValueContainer;

public abstract class AbstractNgramMap<T> implements NgramMap<T>, Serializable
{
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	protected static final byte NUM_BITS_PER_BYTE = Byte.SIZE;

	protected final int NUM_WORD_BITS;

	protected final int NUM_SUFFIX_BITS;

	protected final long WORD_BIT_MASK;

	protected final long SUFFIX_BIT_MASK;

	/**
	 * @param key
	 * @return
	 */
	protected final long contextOffsetOf(final long key) {
		return (key & SUFFIX_BIT_MASK);
	}

	/**
	 * @param key
	 * @return
	 */
	protected final int wordOf(final long key) {
		return (int) ((key & WORD_BIT_MASK) >>> (NUM_SUFFIX_BITS));
	}

	/**
	 * @param word
	 * @param suffixIndex
	 * @return
	 */
	protected final long combineToKey(final int word, final long suffixIndex) {
		final long key = (((long) word) << (NUM_SUFFIX_BITS)) | suffixIndex;
		assert key >= 0 : "Trouble creating key " + word + " :: " + suffixIndex + ". Might need to increase numWordBits.";
		return key;
	}

	protected final ValueContainer<T> values;

	protected final ConfigOptions opts;

	protected AbstractNgramMap(final ValueContainer<T> values, final ConfigOptions opts) {
		this.values = values;
		this.opts = opts;
		this.NUM_WORD_BITS = opts.numWordBits;
		NUM_SUFFIX_BITS = (64 - NUM_WORD_BITS);
		WORD_BIT_MASK = ((1L << NUM_WORD_BITS) - 1) << (NUM_SUFFIX_BITS);
		SUFFIX_BIT_MASK = ((1L << NUM_SUFFIX_BITS) - 1);
	}

	protected static boolean equals(final int[] ngram, final int startPos, final int endPos, final int[] cachedNgram) {
		if (cachedNgram.length != endPos - startPos) return false;
		for (int i = 0; i < endPos - startPos; ++i) {
			if (ngram[startPos + i] != cachedNgram[i]) return false;
		}
		return true;
	}

	protected static int[] getSubArray(final int[] ngram, final int startPos, final int endPos) {
		return Arrays.copyOfRange(ngram, startPos, endPos);

	}

	protected static boolean containsOutOfVocab(final int[] ngram, final int startPos, final int endPos) {
		for (int i = startPos; i < endPos; ++i) {
			if (ngram[i] < 0) return true;
		}
		return false;
	}

	@Override
	public ValueContainer<T> getValues() {
		return values;
	}

}