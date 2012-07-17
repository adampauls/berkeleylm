package edu.berkeley.nlp.lm;

import java.util.List;

import edu.berkeley.nlp.lm.collections.BoundedList;
import edu.berkeley.nlp.lm.util.Annotations.OutputParameter;

/**
 * Interface for language models which expose the internal context-encoding for
 * more efficient queries. (Note: language model implementations may internally
 * use a context-encoding without implementing this interface). A
 * context-encoding encodes an n-gram as a integer representing the last word,
 * and an offset which serves as a logical pointer to the (n-1) prefix words.
 * The integers represent words of type <code>W</code> in the vocabulary, and the mapping
 * from the vocabulary to integers is managed by an instance of the {@link WordIndexer} class.
 * 
 * @author adampauls
 * 
 * @param <W>
 */
public interface ContextEncodedNgramLanguageModel<W> extends NgramLanguageModel<W>
{

	/**
	 * Simple class for returning context offsets
	 * 
	 * @author adampauls
	 * 
	 */
	public static class LmContextInfo
	{

		/**
		 * Offset of context (prefix) of an n-gram
		 */
		public long offset = -1L;

		/**
		 * The (0-based) length of <code>context</code> (i.e.
		 * <code>order == 0</code> iff <code>context</code> refers to a
		 * unigram).
		 * 
		 * Use -1 for an empty context.
		 */
		public int order = -1;

	}

	/**
	 * Get the score for an n-gram, and also get the context offset of the
	 * n-gram's suffix.
	 * 
	 * @param contextOffset
	 *            Offset of context (prefix) of an n-gram
	 * @param contextOrder
	 *            The (0-based) length of <code>context</code> (i.e.
	 *            <code>order == 0</code> iff <code>context</code> refers to a
	 *            unigram).
	 * @param word
	 *            Last word of the n-gram
	 * @param outputContext
	 *            Offset of the suffix of the input n-gram. If the parameter is
	 *            <code>null</code> it will be ignored. This can be passed to
	 *            future queries for efficient access.
	 * @return
	 */
	public float getLogProb(long contextOffset, int contextOrder, int word, @OutputParameter LmContextInfo outputContext);

	/**
	 * Gets the offset which refers to an n-gram. If the n-gram is not in the
	 * model, then it returns the shortest suffix of the n-gram which is. This
	 * operation is not necessarily fast.
	 * 
	 */
	public LmContextInfo getOffsetForNgram(int[] ngram, int startPos, int endPos);

	/**
	 * Gets the n-gram referred to by a context-encoding. This operation is not
	 * necessarily fast.
	 * 
	 */
	public int[] getNgramForOffset(long contextOffset, int contextOrder, int word);

	public static class DefaultImplementations
	{

		public static <T> float scoreSentence(final List<T> sentence, final ContextEncodedNgramLanguageModel<T> lm) {
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

		public static <T> float getLogProb(final List<T> ngram, final ContextEncodedNgramLanguageModel<T> lm) {
			final LmContextInfo contextOutput = new LmContextInfo();
			final WordIndexer<T> wordIndexer = lm.getWordIndexer();
			float score = Float.NaN;
			for (int i = 0; i < ngram.size(); ++i) {
				score = lm.getLogProb(contextOutput.offset, contextOutput.order, wordIndexer.getIndexPossiblyUnk(ngram.get(i)), contextOutput);
			}
			return score;
		}

	}

}
