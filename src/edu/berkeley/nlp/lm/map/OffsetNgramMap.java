package edu.berkeley.nlp.lm.map;

public interface OffsetNgramMap<T> extends NgramMap<T>
{

	public static class ValueOffsetPair<T>
	{

		public T getValue() {
			return value;
		}

		public long getOffset() {
			return offset;
		}

		/**
		 * @param value
		 * @param offset
		 */
		public ValueOffsetPair(final T value, final long offset) {
			super();
			this.value = value;
			this.offset = offset;
		}

		T value;

		long offset;
	}

	public long getOffset(int[] ngram, int startPos, int endPos);

}
