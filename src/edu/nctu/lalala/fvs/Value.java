package edu.nctu.lalala.fvs;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Value {
	double min;
	double max;
	private static final double MIN_VALUE = -3000000000D;
	private static final double MAX_VALUE = 3000000000D;

	public Value(String value) {
		Pattern pattern = Pattern.compile("-?((\\d(\\.)*)+|inf)");
		Matcher m = pattern.matcher(value);
		int counter = 0;
		while (m.find()) {
			String s = m.group();
			// Normalize minus sign
			if (counter == 1)
				s = s.replace("-", "");
			if (s.equals("-inf"))
				s = MIN_VALUE + "";
			else if (s.equals("inf"))
				s = MAX_VALUE + "";
//			System.out.println(s);
			if (counter == 0)
				min = Double.parseDouble(s);
			else if (counter == 1) {
				max = Double.parseDouble(s);
				break;
			}
			counter++;
		}
	}

	public Value(double value) {
		this.min = value;
		this.max = value;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		final Value other = (Value) obj;
		if ((this.min <= other.min) && (this.max >= other.max) || (other.min <= this.min) && (other.max >= this.max))
			return true;
		else
			return false;
	}
	
	@Override
	public int hashCode() {
		return 1;
	}

	@Override
	public String toString() {
		return String.format("<%.2f,%.2f>", min, max);
	}
}
