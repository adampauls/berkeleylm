package edu.berkeley.nlp.lm.io;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import edu.berkeley.nlp.lm.map.NgramMap;
import edu.berkeley.nlp.lm.util.Logger;

/**
 * Reader callback which adds n-grams to an NgramMap
 * 
 * @author adampauls
 * 
 * @param <V>
 *            Value type
 */
public final class NgramMapAddingCallback<V> implements ArpaLmReaderCallback<V>
{

	private final NgramMap<V> map;

	//	int warnCount = 0;

	private final List<int[]> failures;

	private final boolean canFail;

	public NgramMapAddingCallback(final NgramMap<V> map, List<int[]> failures) {
		this.map = map;
		this.canFail = failures == null;
		this.failures = canFail ? new ArrayList<int[]>() : failures;
	}

	@Override
	public void call(final int[] ngram, int startPos, int endPos, final V v, final String words) {
		final long add = map.put(ngram, startPos, endPos, v);

		if (add < 0) {
			if (canFail) {
				for (int endPos_ = endPos - 1; (endPos_ > startPos); endPos_--) {
					if (!map.contains(ngram, startPos, endPos_)) {
						failures.add(Arrays.copyOfRange(ngram, startPos, endPos_));
					}
				}
				for (int startPos_ = startPos; (startPos_ < endPos); startPos_++) {
					if (!map.contains(ngram, startPos + 1, endPos)) {
						failures.add(Arrays.copyOfRange(ngram, startPos_, endPos));
					}
				}
			} else {
				throw new RuntimeException("Failed to add line " + words);
			}
		}
	}

	@Override
	public void handleNgramOrderFinished(final int order) {
		map.handleNgramsFinished(order);
		for (int[] ngram : failures) {
			if (ngram.length == order + 1) {
				map.put(ngram, 0, ngram.length, null);
			}
		}
	}

	@Override
	public void cleanup() {
		if (failures.isEmpty() || !canFail) map.trim();
	}

	@Override
	public void initWithLengths(final List<Long> numNGrams) {
		map.initWithLengths(numNGrams);
	}

	public List<int[]> getFailures() {
		return failures;
	}

}