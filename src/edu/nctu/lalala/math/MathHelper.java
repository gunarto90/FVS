package edu.nctu.lalala.math;

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
		Double[] q = new Double[3];	// Q1, Q2, Q3
		final int QUARTILE = 4;
		for(int i=1; i<QUARTILE; i++)
		{
			int pos = (data.length* i / QUARTILE) + 1;
			q[i-1] = data[pos];
		}
		return q;
	}
	
	
}
