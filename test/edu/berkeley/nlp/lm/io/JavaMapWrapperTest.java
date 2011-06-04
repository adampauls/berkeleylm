package edu.berkeley.nlp.lm.io;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.map.JavaMapWrapper;
import edu.berkeley.nlp.lm.util.LongRef;

public class JavaMapWrapperTest
{
	@Test
	public void testBothMapWrapper() {
		StupidBackoffLm<String> lm = LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), false);
		StupidBackoffLm<String> lm2 = LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), true);

		for (int order = 0; order < lm.getLmOrder(); ++order) {
			final JavaMapWrapper<String, LongRef> map = new JavaMapWrapper<String, LongRef>(lm.getNgramMap(), lm.getWordIndexer(), order);
			final JavaMapWrapper<String, LongRef> map2 = new JavaMapWrapper<String, LongRef>(lm2.getNgramMap(), lm2.getWordIndexer(), order);
			if (order == 2) Assert.assertEquals(map.get(Arrays.asList(",", "the", "(")).value, 50);
			if (order == 2) Assert.assertTrue(map.containsKey(Arrays.asList(",", "the", "(")));
			if (order == 2) Assert.assertFalse(map.containsKey(Arrays.asList("the", "the", "(")));
			if (order == 0) Assert.assertEquals(map.get(Arrays.asList("the")).value, 19401194714L);
			if (order == 2) Assert.assertEquals(map2.get(Arrays.asList(",", "the", "(")).value, 50);
			if (order == 0) Assert.assertEquals(map2.get(Arrays.asList("the")).value, 19401194714L);
			Assert.assertEquals(map.size(), map2.size());
			for (Entry<List<String>, LongRef> entry : map.entrySet()) {
				Assert.assertEquals(map2.get(entry.getKey()), entry.getValue());
			}
			for (Entry<List<String>, LongRef> entry : map2.entrySet()) {
				Assert.assertEquals(map.get(entry.getKey()), entry.getValue());
			}

		}
	}
}
