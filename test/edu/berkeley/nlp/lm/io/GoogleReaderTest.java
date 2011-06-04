package edu.berkeley.nlp.lm.io;

import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.StupidBackoffLm;

public class GoogleReaderTest
{
	@Test
	public void testHash() {
		StupidBackoffLm<String> lm = LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), false);
		checkScores(lm);
	}
	
	@Test
	public void testCompressed() {
		StupidBackoffLm<String> lm = LmReaders.readLmFromGoogleNgramDir(FileUtils.getFile("googledir").getPath(), true);
		checkScores(lm);
	}

	/**
	 * @param lm
	 */
	private void checkScores(StupidBackoffLm<String> lm) {
		Assert.assertEquals(lm.getLogProb(Arrays.asList("the", "(")), -12.314105, 1e-3);
		Assert.assertEquals(lm.getLogProb(Arrays.asList("of", "the", "(")), -6.684612, 1e-3);
		Assert.assertEquals(lm.getLogProb(Arrays.asList("a", "the", "(")), -13.230395, 1e-3);
		Assert.assertEquals(lm.getLogProb(Arrays.asList("a", ")", "(")), -5.6564045, 1e-3);
		Assert.assertEquals(lm.getLogProb(Arrays.asList("the", "of", "a")), -15.491532, 1e-3);
	}

	
}
