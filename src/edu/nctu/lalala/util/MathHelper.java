package edu.nctu.lalala.util;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MathHelper {
	private MathHelper() {
	}

	private static final MathHelper singleton = new MathHelper();

	public static MathHelper getInstance() {
		return singleton;
	}
	
	/**
	 * Calculate average of a list
	 * @param data
	 * @return
	 */
	public Double calculateAverage(Double... data)
	{
		Double mean = 0.0;
		if (data.length > 0) {
			for (Double x : data) {
				mean += x;
			}
			mean = mean / data.length;
		}
		return mean;
	}
	
	/**
	 * Calculate stdev of a list
	 * 
	 * @param inst
	 * @param average
	 * @return
	 */
	public Double calculateStdev(Double average, Double... data) {
		Double stdev = 0.0;
		if (data.length > 0) {
			for (int i = 0; i < data.length; i++) {
				stdev += Math.pow(data[i] - average, 2);
			}
			stdev /= data.length;
		}
		return stdev;
	}
	
	/**
	 * Calculate quartile of a list
	 * @param data
	 * @return
	 */
	public Double[] calculateQuartile(Double... data)
	{
		List<Double> temp = Arrays.asList(data);
		Collections.sort(temp);
		Double[] q = new Double[3];	// Q1, Q2, Q3
		final int QUARTILE = 4;
		for(int i=1; i<QUARTILE; i++)
		{
			int pos = (temp.size()* i / QUARTILE) + 1;
			q[i-1] = temp.get(pos);
		}
		return q;
	}
	
	public double calculateEntropy(double[] counter, double frequency) {
		if (frequency == 0)
			return 0;
		double entropy = 0;
		double[] p = new double[counter.length];
		for (int i = 0; i < counter.length; i++) {
			p[i] = counter[i] / frequency;
		}
		for (int i = 0; i < p.length; i++) {
			if (p[i] == 0.0F)
				continue;
			entropy -= p[i] * (Math.log(p[i]) / Math.log(p.length));
		}
		return entropy;
	}
	
}
