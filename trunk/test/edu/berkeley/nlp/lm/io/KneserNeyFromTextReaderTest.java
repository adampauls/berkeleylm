package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

public class KneserNeyFromTextReaderTest
{

	@Test
	public void myTest() {
		final StringWordIndexer wordIndexer = new StringWordIndexer();
		wordIndexer.setStartSymbol("<s>");
		wordIndexer.setEndSymbol("</s>");
		wordIndexer.setUnkSymbol("<unk>");
		File txtFile = getFile("tiny_test.txt");
		File arpaFile = getFile("tiny_test.arpa");
		HashNgramMap<KneserNeyCounts> countNgrams = KneserNeyFromTextReader.readFromFiles(Arrays.asList(txtFile), wordIndexer, 3);
		final StringWriter stringWriter = new StringWriter();
		KneserNeyFromTextReader.writeToPrintWriter(wordIndexer, countNgrams, new PrintWriter(stringWriter));
		System.out.println(stringWriter.toString());
		List<String> arpaLines = Arrays.asList(stringWriter.toString().split("\n"));
		Collections.sort(arpaLines);
	}

	/**
	 * @param testFileName
	 * @return
	 */
	private File getFile(final String testFileName) {
		File txtFile = null;
		try {
			txtFile = new File(KneserNeyFromTextReaderTest.class.getResource(testFileName).toURI());
		} catch (URISyntaxException e) {
			Assert.fail(e.toString());
		}
		Assert.assertNotNull("Could not read " + testFileName, txtFile);
		return txtFile;
	}

}
