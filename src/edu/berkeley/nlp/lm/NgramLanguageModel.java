package edu.berkeley.nlp.lm;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import edu.berkeley.nlp.lm.collections.Counter;

/**
 * 
 * Base interface for an n-gram language model, which exposes only inefficient
 * convenience methods. See {@link ContextEncodedNgramLanguageModel} and
 * {@link ArrayEncodedNgramLanguageModel} for more efficient accessors.
 * 
 * @author adampauls
 * 
 * @param <W>
 * 
 */
public interface NgramLanguageModel<W>
{

	/**
	 * Maximum size of n-grams stored by the model.
	 * 
	 * @return
	 */
	public int getLmOrder();

	/**
	 * Each LM must have a WordIndexer which assigns integer IDs to each word W
	 * in the language.
	 * 
	 * @return
	 */
	public WordIndexer<W> getWordIndexer();

	/**
	 * Scores a complete sentence, taking appropriate care with the start- and
	 * end-of-sentence symbols. This is a convenience method and will generally
	 * be inefficient.
	 * 
	 * @return
	 */
	public float scoreSentence(List<W> sentence);

	/**
	 * 
	 * Scores an n-gram. This is a convenience method and will generally be
	 * relatively inefficient. More efficient versions are available in
	 * {@link ArrayEncodedNgramLanguageModel#getLogProb(int[], int, int)} and
	 * {@link ContextEncodedNgramLanguageModel#getLogProb(long, int, int, edu.berkeley.nlp.lm.ContextEncodedNgramLanguageModel.LmContextInfo)}
	 * .
	 */
	public float getLogProb(List<W> ngram);

	/**
	 * Sets the (log) probability for an OOV word. Note that this is in general
	 * different from the log prob of the <code>unk</code> tag probability.
	 * 
	 * @author adampauls
	 * 
	 */
	public void setOovWordLogProb(float logProb);

	public static class StaticMethods
	{

		public static <T> int[] toIntArray(final List<T> ngram, final ArrayEncodedNgramLanguageModel<T> lm) {
			final int[] ints = new int[ngram.size()];
			final WordIndexer<T> wordIndexer = lm.getWordIndexer();
			for (int i = 0; i < ngram.size(); ++i) {
				ints[i] = wordIndexer.getIndexPossiblyUnk(ngram.get(i));
			}
			return ints;
		}

		public static <T> List<T> toObjectList(final int[] ngram, final ArrayEncodedNgramLanguageModel<T> lm) {
			final List<T> ret = new ArrayList<T>(ngram.length);
			final WordIndexer<T> wordIndexer = lm.getWordIndexer();
			for (int i = 0; i < ngram.length; ++i) {
				ret.add(wordIndexer.getWord(ngram[i]));
			}
			return ret;
		}

		/**
		 * Samples from this language model. This is not meant to be
		 * particularly efficient
		 * 
		 * @param random
		 * @return
		 */
		public static <W> List<W> sample(Random random, final NgramLanguageModel<W> lm) {
			return sample(random, lm, 1.0);
		}

		public static <W> List<W> sample(Random random, final NgramLanguageModel<W> lm, final double sampleTemperature) {
			List<W> ret = new ArrayList<W>();
			ret.add(lm.getWordIndexer().getStartSymbol());
			while (true) {
				final int contextEnd = ret.size();
				final int contextStart = Math.max(0, contextEnd - lm.getLmOrder() + 1);
				Counter<W> c = new Counter<W>();
				List<W> ngram = new ArrayList<W>(ret.subList(contextStart, contextEnd));
				ngram.add(null);
				for (int index = 0; index < lm.getWordIndexer().numWords(); ++index) {

					W word = lm.getWordIndexer().getWord(index);
					if (word.equals(lm.getWordIndexer().getStartSymbol())) continue;
					if (ret.size() <= 1 && word.equals(lm.getWordIndexer().getEndSymbol())) continue;

					ngram.set(ngram.size() - 1, word);
					c.setCount(word, Math.exp(sampleTemperature * lm.getLogProb(ngram) * Math.log(10)));
				}
				W sample = c.sample(random);
				ret.add(sample);
				if (sample.equals(lm.getWordIndexer().getEndSymbol())) break;

			}
			return ret.subList(1, ret.size() - 1);
		}

		/**
		 * Builds a distribution over next possible words given the context. Context can be of any length, but 
		 * only at most <code>lm.getLmOrder() - 1</code> words are actually used.
		 * 
		 * @param <W>
		 * @param lm
		 * @param context
		 * @return
		 */
		public static <W> Counter<W> getDistributionOverNextWords(final NgramLanguageModel<W> lm, List<W> context) {
			List<W> ngram = new ArrayList<W>();
			for (int i = 0; i < lm.getLmOrder() - 1 && i < context.size(); ++i) {
				ngram.add(context.get(context.size() - i - 1));
			}
			if (ngram.size() < lm.getLmOrder() - 1) ngram.add(lm.getWordIndexer().getStartSymbol());
			Collections.reverse(ngram);
			ngram.add(null);
			Counter<W> c = new Counter<W>();
			for (int index = 0; index < lm.getWordIndexer().numWords(); ++index) {
				W word = lm.getWordIndexer().getWord(index);
				if (word.equals(lm.getWordIndexer().getStartSymbol())) continue;
				ngram.set(ngram.size() - 1, word);
				c.setCount(word, Math.exp(lm.getLogProb(ngram) * Math.log(10)));
			}
			return c;
		}

	}

}
