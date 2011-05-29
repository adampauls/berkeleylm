package edu.berkeley.nlp.lm.io;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.Arrays;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

public class KneserNeyFromTextReaderTest
{
	@Test
	public void myTest() {
		String[] lines = new String[] { "This is a test sentence.", "This is another test sentence." };
		final StringWordIndexer wordIndexer = new StringWordIndexer();
		wordIndexer.setStartSymbol("<s>");
		wordIndexer.setEndSymbol("</s>");
		wordIndexer.setUnkSymbol("<unk>");
		HashNgramMap<KneserNeyCounts> countNgrams = KneserNeyFromTextReader.countNgrams(wordIndexer, 3, Arrays.asList(lines));
		final StringWriter stringWriter = new StringWriter();
		KneserNeyFromTextReader.writeToPrintWriter(wordIndexer, countNgrams, new PrintWriter(stringWriter));
		System.out.println(stringWriter.toString());
	}

}
