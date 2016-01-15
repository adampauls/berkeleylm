package edu.berkeley.nlp.lm;

import java.util.List;

import edu.berkeley.nlp.lm.collections.BoundedList;

/**
 * Top-level interface for an n-gram language model which accepts n-gram in an
 * array-of-integers encoding. The integers represent words of type
 * <code>W</code> in the vocabulary, and the mapping from the vocabulary to
 * integers is managed by an instance of the {@link WordIndexer} class.
 * 
 * @author adampauls
 */
public interface ArrayEncodedNgramLanguageModel<W> extends NgramLanguageModel<W>
{

	/**
	 * Calculate language model score of an n-gram. <b>Warning:</b> if you
	 * pass in an n-gram of length greater than <code>getLmOrder()</code>,
	 * this call will silently ignore the extra words of context. In other
	 * words, if you pass in a 5-gram (<code>endPos-startPos == 5</code>) to
	 * a 3-gram model, it will only score the words from <code>startPos + 2</code>
	 * to <code>endPos</code>.
	 * 
	 * @param ngram
	 *            array of words in integer representation
	 * @param startPos
	 *            start of the portion of the array to be read
	 * @param endPos
	 *            end of the portion of the array to be read.
	 * @return
	 */
	public float getLogProb(int[] ngram, int startPos, int endPos);

	/**
	 * Equivalent to <code>getLogProb(ngram, 0, ngram.length)</code>
	 * 
	 * @see #getLogProb(int[], int, int)
	 */
	public float getLogProb(int[] ngram);

	public static class DefaultImplementations
	{

		public static <T> float scoreSentence(final List<T> sentence, final ArrayEncodedNgramLanguageModel<T> lm) {
			final List<T> sentenceWithBounds = new BoundedList<T>(sentence, lm.getWordIndexer().getStartSymbol(), lm.getWordIndexer().getEndSymbol());

			final int lmOrder = lm.getLmOrder();
			float sentenceScore = 0.0f;
			for (int i = 1; i < lmOrder - 1 && i <= sentenceWithBounds.size() + 1; ++i) {
				final List<T> ngram = sentenceWithBounds.subList(-1, i);
				final float scoreNgram = lm.getLogProb(ngram);
				sentenceScore += scoreNgram;
			}
			for (int i = lmOrder - 1; i < sentenceWithBounds.size() + 2; ++i) {
				final List<T> ngram = sentenceWithBounds.subList(i - lmOrder, i);
				final float scoreNgram = lm.getLogProb(ngram);
				sentenceScore += scoreNgram;
			}
			return sentenceScore;
		}

		public static <T> float getLogProb(final int[] ngram, final ArrayEncodedNgramLanguageModel<T> lm) {
			return lm.getLogProb(ngram, 0, ngram.length);
		}

		public static <T> float getLogProb(final List<T> ngram, final ArrayEncodedNgramLanguageModel<T> lm) {
			final int[] ints = StaticMethods.toIntArray(ngram, lm);
			return lm.getLogProb(ints, 0, ints.length);

		}
	}

}
