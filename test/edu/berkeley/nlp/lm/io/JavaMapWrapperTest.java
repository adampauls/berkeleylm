package edu.berkeley.nlp.lm.io;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.StupidBackoffLm;
import edu.berkeley.nlp.lm.map.JavaMapWrapper;
import edu.berkeley.nlp.lm.util.LongRef;

public class JavaMapWrapperTest
{
	@Test
	public void testHashMapWrapper() {
		StupidBackoffLm<String> lm = LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), false);
		for (int order = 0; order < lm.getLmOrder(); ++order) {
			System.out.println(new JavaMapWrapper<String, LongRef>(lm.getNgramMap(), lm.getWordIndexer(), order));
		}
	}

	@Test
	public void testCompressedMapWrapper() {
		StupidBackoffLm<String> lm = LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), true);
		for (int order = 0; order < lm.getLmOrder(); ++order) {
			System.out.println(new JavaMapWrapper<String, LongRef>(lm.getNgramMap(), lm.getWordIndexer(), order));
		}
	}

}
