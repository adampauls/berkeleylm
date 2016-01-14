package edu.berkeley.nlp.lm.phrasetable;

import java.io.File;
import java.net.URISyntaxException;
import java.net.URL;

import org.junit.Assert;

public class FileUtils
{

	/**
	 * @param testFileName
	 * @return
	 */
	public static File getFile(final String testFileName) {
		File txtFile = null;
		try {
			final URL resource = FileUtils.class.getResource(testFileName);
			txtFile = new File(resource.toURI());
		} catch (final URISyntaxException e) {
			Assert.fail(e.toString());
		}
		Assert.assertNotNull("Could not read " + testFileName, txtFile);
		return txtFile;
	}

}
