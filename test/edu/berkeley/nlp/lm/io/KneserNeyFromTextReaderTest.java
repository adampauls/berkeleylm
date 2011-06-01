package edu.berkeley.nlp.lm.io;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import edu.berkeley.nlp.lm.StringWordIndexer;
import edu.berkeley.nlp.lm.collections.Iterators;
import edu.berkeley.nlp.lm.map.HashNgramMap;
import edu.berkeley.nlp.lm.util.Pair;
import edu.berkeley.nlp.lm.values.KneseryNeyCountValueContainer.KneserNeyCounts;

public class KneserNeyFromTextReaderTest
{

	@Test
	public void testBigram() {
		doTest("tiny_test_bigram", new float[] { 0.75f, 0.33333f });
	}

	@Test
	public void testTrigram() {
		doTest("tiny_test_trigram", new float[] { 0.75f, 0.6f, 0.6f });
	}

	@Test
	public void testFivegram() {
		doTest("tiny_test_fivegram", new float[] { 0.4f, 0.5f, 0.5f, 0.538462f, 0.454545f });
	}

	@Test
	public void testBig() {
		doTest("big_test", new float[] { 0.755639f, 0.891934f, 0.944268f, 0.955941f, 0.359436f });
	}

	/**
	 * @param prefix
	 * @param order
	 * @param discounts
	 */
	private void doTest(String prefix, final float[] discounts) {
		final StringWordIndexer wordIndexer = new StringWordIndexer();
		int order = discounts.length;
		wordIndexer.setStartSymbol("<s>");
		wordIndexer.setEndSymbol("</s>");
		wordIndexer.setUnkSymbol("<unk>");
		File txtFile = FileUtils.getFile(prefix + ".txt");
		File goldArpaFile = FileUtils.getFile(prefix + ".arpa");
		StringWriter stringWriter = new StringWriter();
		final KneserNeyFromTextReader<String> reader = new KneserNeyFromTextReader<String>(Arrays.asList(txtFile), wordIndexer, order);
		reader.parse(new KneserNeyLmReaderCallback<String>(new PrintWriter(stringWriter), wordIndexer, order, discounts));
		List<String> arpaLines = new ArrayList<String>(Arrays.asList(stringWriter.toString().split("\n")));
		sortAndRemoveBlankLines(arpaLines);
		List<String> goldArpaLines = getLines(goldArpaFile);
		sortAndRemoveBlankLines(goldArpaLines);
		compareLines(arpaLines, goldArpaLines);
	}

	/**
	 * @param arpaLines
	 * @param goldArpaLines
	 */
	private void compareLines(List<String> arpaLines, List<String> goldArpaLines) {
		Assert.assertEquals(arpaLines.size(), goldArpaLines.size());
		for (Pair<String, String> lines : Iterators.able(Iterators.zip(arpaLines.iterator(), goldArpaLines.iterator()))) {
			String testLine = lines.getFirst();
			String goldLine = lines.getSecond();
			if (goldLine.startsWith("-")) {
				Assert.assertTrue(lines.toString(), testLine.startsWith("-"));
				String[] testSplit = testLine.split("\t");
				String[] goldSplit = goldLine.split("\t");
				Assert.assertEquals(lines.toString(), testSplit.length, goldSplit.length);
				Assert.assertTrue(lines.toString(), testSplit.length == 2 || testSplit.length == 3);
				Assert.assertEquals(lines.toString(), testSplit[1], goldSplit[1]);
				if (!testSplit[1].startsWith("<s>")) {
					// SRILM appears to do the wrong thing with the <s> start tag, so we don't test for equality
					Assert.assertEquals(lines.toString(), Double.parseDouble(testSplit[0]), Double.parseDouble(goldSplit[0]), 1e-3);
					if (testSplit.length == 3) {
						Assert.assertEquals(lines.toString(), Double.parseDouble(testSplit[2]), Double.parseDouble(goldSplit[2]), 1e-3);
					}
				}

			} else {
				Assert.assertEquals(testLine, goldLine);
			}
		}
	}

	private List<String> getLines(File goldArpaFile) {
		List<String> ret = new ArrayList<String>();
		try {
			for (String line : Iterators.able(IOUtils.lineIterator(goldArpaFile.getAbsolutePath()))) {
				ret.add(line);
			}
			return ret;
		} catch (IOException e) {
			throw new RuntimeException(e);

		}
	}

	/**
	 * @param arpaLines
	 */
	private void sortAndRemoveBlankLines(List<String> arpaLines) {
		Collections.sort(arpaLines, new Comparator<String>()
		{

			@Override
			public int compare(String arg0, String arg1) {
				String[] split1 = arg0.split("\t");
				String[] split2 = arg1.split("\t");
				int x = Double.compare(split1.length, split2.length);
				if (x != 0) return x;
				if (split1.length > 1) return split1[1].compareTo(split2[1]);
				return split1[0].compareTo(split2[0]);
			}
		});
		for (int i = arpaLines.size() - 1; i >= 0; i--) {
			if (arpaLines.get(i).trim().isEmpty()) arpaLines.remove(i);
		}
	}

}
