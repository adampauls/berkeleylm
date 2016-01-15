package edu.berkeley.nlp.lm.phrasetable;

import java.util.Arrays;
import java.util.List;

import junit.framework.Assert;

import org.junit.Test;

import edu.berkeley.nlp.lm.WordIndexer;
import edu.berkeley.nlp.lm.phrasetable.MosesPhraseTable.TargetSideTranslation;

public class MosesPhraseTableTest
{

	@Test
	public void testPhraseTable() {
		final MosesPhraseTable readFromFile = MosesPhraseTable.readFromFile(FileUtils.getFile("test_phrase_table.moses").getPath());
		{
			final int[] array1 = WordIndexer.StaticMethods.toArrayFromStrings(readFromFile.getWordIndexer(), Arrays.asList("i", "like"));
			final List<TargetSideTranslation> translations = readFromFile.getTranslations(array1, 0, array1.length);
			Assert.assertEquals(3, translations.size());
			Assert.assertEquals(1, translations.get(2).trgWords.length);
			Assert.assertEquals(2, translations.get(0).trgWords.length);
		}

		{
			final int[] array1 = WordIndexer.StaticMethods.toArrayFromStrings(readFromFile.getWordIndexer(), Arrays.asList("i"));
			final List<TargetSideTranslation> translations = readFromFile.getTranslations(array1, 0, array1.length);
			Assert.assertEquals(1, translations.size());
			Assert.assertEquals(1, translations.get(0).trgWords.length);
		}

		{
			final int[] array1 = WordIndexer.StaticMethods.toArrayFromStrings(readFromFile.getWordIndexer(), Arrays.asList("want"));
			final List<TargetSideTranslation> translations = readFromFile.getTranslations(array1, 0, array1.length);
			Assert.assertEquals(0, translations.size());
		}
	}

	public static void main(final String[] argv) {
		new MosesPhraseTableTest().testPhraseTable();
	}
}
