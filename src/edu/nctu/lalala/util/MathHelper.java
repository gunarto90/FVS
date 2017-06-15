package edu.nctu.lalala.util;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.inference.ChiSquareTest;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.main.Main;

public class MathHelper {
	private MathHelper() {
	}

	private static final MathHelper singleton = new MathHelper();

	public static MathHelper getInstance() {
		return singleton;
	}

	/**
	 * Calculate average of a list
	 * 
	 * @param data
	 * @return
	 */
	public Double calculateAverage(Double... data) {
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
	 * 
	 * @param data
	 * @return
	 */
	public Double[] calculateQuartile(Double... data) {
		List<Double> temp = Arrays.asList(data);
		Collections.sort(temp);
		Double[] q = new Double[3]; // Q1, Q2, Q3
		final int QUARTILE = 4;
		for (int i = 1; i < QUARTILE; i++) {
			int pos = (temp.size() * i / QUARTILE) + 1;
			q[i - 1] = temp.get(pos);
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
			// entropy -= p[i] * (Math.log(p[i]) / Math.log(p.length));
			entropy -= p[i] * (Math.log(p[i]) / Main.NUMBER_OF_CLASS);
		}
		return entropy;
	}

	public Map<FV, Double> calculateChiSquare(Map<FV, Collection<FV>> fv_list, int numOfClasses) {
		double verysmallnumber = 0.0000000001;
		Map<FV, Double> result = new HashMap<FV, Double>();

		Map<FV, long[]> dist_set = new HashMap<FV, long[]>();
		Map<FV, double[]> expect_set = new HashMap<FV, double[]>();
		Map<FV, Integer> count_set = new HashMap<FV, Integer>();
		int[] all_activities = new int[numOfClasses];

		for (Entry<FV, Collection<FV>> entry : fv_list.entrySet()) {
			for (FV fv : entry.getValue()) {
				long[] dist = dist_set.getOrDefault(fv, new long[numOfClasses]);
				int label = (int) fv.getLabel();
				dist[label] += 1;
				all_activities[label] += 1;
				count_set.put(fv, count_set.getOrDefault(fv, 0) + 1);
				dist_set.put(fv, dist);
			}
			// System.out.println(entry.getKey());
			// System.out.println(count_set.get(entry.getKey()));
		}

		// System.out.println(Arrays.toString(all_activities));
		int sum = IntStream.of(all_activities).sum();
		for (Entry<FV, Collection<FV>> entry : fv_list.entrySet()) {
			double[] expect_dist = new double[numOfClasses];
			for (int i = 0; i < numOfClasses; i++) {
				// expect_dist[i] = (double) all_activities[i] *
				// LongStream.of(dist_set.get(entry.getKey())).sum() / sum;
				expect_dist[i] = (double) all_activities[i] * count_set.get(entry.getKey()) / sum;
				if (expect_dist[i] <= 0)
					expect_dist[i] = verysmallnumber;
			}
			expect_set.put(entry.getKey(), expect_dist);
			// System.out.println(Arrays.toString(dist_set.get(entry.getKey())));
			// System.out.println(Arrays.toString(expect_set.get(entry.getKey())));
		}

		ChiSquareTest chi = new ChiSquareTest();
		for (Entry<FV, Collection<FV>> entry : fv_list.entrySet()) {
			FV fv = entry.getKey();
			result.put(fv, chi.chiSquare(expect_set.get(fv), dist_set.get(fv)));
		}

		return result;
	}

	public int[] getMembership(double[] arr, int size) {
		int[] membership = new int[size];
		for (double d : arr) {
			int x = (int) (Math.floor(d / 0.1));
			if (x == size)
				x = size - 1;
			membership[x] += 1;
		}
		return membership;
	}

}
