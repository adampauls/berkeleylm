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

	private static class TestInfo
	{
		String prefix;

		int lmOrder;

		float[] discounts;

		/**
		 * @param prefix
		 * @param lmOrder
		 * @param discounts
		 */
		public TestInfo(String prefix, int lmOrder, float[] discounts) {
			super();
			this.prefix = prefix;
			this.lmOrder = lmOrder;
			this.discounts = discounts;
		}

	}

	@Test
	public void test() {

		TestInfo[] tests = new TestInfo[3];

		int k = 0;
		tests[k++] = new TestInfo("tiny_test_bigram", 2, new float[] { 0.75f, 0.33333f });
		tests[k++] = new TestInfo("tiny_test_trigram", 3, new float[] { 0.75f, 0.6f, 0.6f });
		tests[k++] = new TestInfo("tiny_test_fivegram", 5, new float[] { 0.75f, 0.75f, 0.75f, 0.77778f, 0.3333333f });
		for (TestInfo fileInfo : tests) {
			String prefix = fileInfo.prefix;
			int order = fileInfo.lmOrder;
			final StringWordIndexer wordIndexer = new StringWordIndexer();
			wordIndexer.setStartSymbol("<s>");
			wordIndexer.setEndSymbol("</s>");
			wordIndexer.setUnkSymbol("<unk>");
			File txtFile = getFile(prefix + ".txt");
			File goldArpaFile = getFile(prefix + ".arpa");
			final KneserNeyFromTextReader<String> reader = new KneserNeyFromTextReader<String>(wordIndexer, order, fileInfo.discounts);
			StringWriter stringWriter = new StringWriter();
			reader.readFromFiles(Arrays.asList(txtFile), new PrintWriter(stringWriter));
			System.out.println(stringWriter.toString());
			List<String> arpaLines = new ArrayList<String>(Arrays.asList(stringWriter.toString().split("\n")));
			sortAndRemoveBlankLines(arpaLines);
			List<String> goldArpaLines = getLines(goldArpaFile);
			sortAndRemoveBlankLines(goldArpaLines);
			compareLines(arpaLines, goldArpaLines);
		}
	}

	/**
	 * @param arpaLines
	 * @param goldArpaLines
	 */
	private void compareLines(List<String> arpaLines, List<String> goldArpaLines) {
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
				Assert.assertEquals(lines.toString(), Double.parseDouble(testSplit[0]), Double.parseDouble(goldSplit[0]), 1e-3);
				if (testSplit.length == 3) {
					Assert.assertEquals(lines.toString(), Double.parseDouble(testSplit[2]), Double.parseDouble(goldSplit[2]), 1e-3);
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
