package edu.nctu.lalala.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.CorrelationMatrix;
import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.Value;
import weka.core.Instance;
import weka.core.Instances;

public class FVSHelper {
	private FVSHelper() {
	}

	private static final FVSHelper singleton = new FVSHelper();

	public static FVSHelper getInstance() {
		return singleton;
	}

	public Map<FV, Collection<FV>> extractValuesFromData(Instances inst) {
		Multimap<FV, FV> fv_list = ArrayListMultimap.create();
		// Instances outFormat = getOutputFormat();
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance ins = inst.instance(i);
			// Skip the class label
			for (int x = 0; x < ins.numAttributes() - 1; x++) {
				Object value = null;
				try {
					value = ins.stringValue(x);
				} catch (Exception e) {
					value = ins.value(x);
				}
				FV fv = new FV(x, value, ins.classValue());
				fv.setNumLabels(inst.numClasses());
				if (!fv_list.put(fv, fv)) {
					System.err.println("Couldn't put duplicates: " + fv);
				}
			}
		}
		Map<FV, Collection<FV>> original_map = fv_list.asMap();
		return original_map;
	}

	public CorrelationMatrix generateCorrelationMatrix(Instances inst) {
		CorrelationMatrix result = new CorrelationMatrix();
		Double[][] CM = new Double[inst.numAttributes()][inst.numAttributes()];
		Double[] average = calculateAverage(inst);
		for (int i = 0; i < inst.numAttributes(); i++) {
			for (int j = 0; j < inst.numAttributes(); j++) {
				CM[i][j] = 0.0;
			}
		}
		// Update CM values
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			for (int j = i + 1; j < inst.numAttributes() - 1; j++) {
				CM[i][j] = calculateLinearCorrelation(inst, average, i, j);
				CM[j][i] = CM[i][j]; // Correlation between x and y is the
										// same with y and x
				if (CM[i][j] != 0)
					result.getCorrValues().add(CM[i][j]);
			}
		}
		Collections.sort(result.getCorrValues());
		result.setCM(CM);
		return result;
	}

	/**
	 * Calculate average of every columns
	 * 
	 * @param inst
	 * @return
	 */
	public Double[] calculateAverage(Instances inst) {
		Double[] average = new Double[inst.numAttributes() - 1];
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			average[i] = 0.0;
		}
		for (int i = 0; i < inst.numInstances(); i++) {
			for (int x = 0; x < inst.instance(i).numAttributes() - 1; x++) {
				Instance ins = inst.instance(i);
				if (ins != null && !Double.isNaN(ins.value(x)))
					average[x] += ins.value(x);
			}
		}
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			average[i] /= inst.numInstances();
		}
		return average;
	}

	/**
	 * Calculate linear correlation between 2 columns Reference:
	 * https://en.wikipedia.org/wiki/Pearson_product-
	 * moment_correlation_coefficient
	 * 
	 * @param inst
	 * @param average
	 * @param x
	 *            Column 1
	 * @param y
	 *            Column 2
	 * @return
	 */
	public Double calculateLinearCorrelation(Instances inst, Double[] average, int x, int y) {
		double corr = 0;
		double top, rootXiXbar, rootYiYbar, bot;
		top = rootXiXbar = rootYiYbar = 0;
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance ins = inst.instance(i);
			if (ins != null && !Double.isNaN(ins.value(x)) && !Double.isNaN(ins.value(y))) {
				top += (ins.value(x) - average[x]) * (ins.value(y) - average[y]);
				rootXiXbar += Math.pow(ins.value(x) - average[x], 2);
				rootYiYbar += Math.pow(ins.value(y) - average[y], 2);
			}
		}
		rootXiXbar = Math.sqrt(rootXiXbar);
		rootYiYbar = Math.sqrt(rootYiYbar);
		bot = rootXiXbar * rootYiYbar;
		if (bot != 0) {
			corr = top / bot;
		}
		return corr;
	}

	public Instances transformInstances(Instances inst, Instances output, Map<FV, Collection<FV>> map) {
		Set<FV> set = map.keySet();
		Double[] substitution = calculateAverage(inst);
		// Prepare the list
		// First level indicate which attribute the FVS resides in
		List<List<Value>> list = new ArrayList<>();
		for (int i = 0; i < inst.numAttributes(); i++) {
			list.add(new ArrayList<Value>());
		}
		// Build the data structure
		for (FV fv : set) {
			list.get(fv.getFeature()).add(new Value(fv.getValue().toString()));
		}
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance instance = getFVSFilteredInstance(output, inst.instance(i), list, substitution);
			output.add(instance);
		}
		return output;
	}

	public Instance getFVSFilteredInstance(Instances output, Instance old_inst, List<List<Value>> list,
			Double[] substitution) {
		double[] oldValues = old_inst.toDoubleArray();
		Instance instance = new Instance(old_inst);
		// Change with value that is available
		for (int i = 0; i < oldValues.length - 1; i++) {
			// System.out.println(oldValues[i]);
			// System.out.println(list.get(i));
			// System.out.println("############################");
			// If list doesn't contain, then delete
			Value v = new Value(oldValues[i]);
			int idx = list.get(i).indexOf(v);
			// If not found in the index
			if (idx == -1) {
				// Change with substitution
				instance.setValue(i, substitution[i]);
				// Change into missing
				// instance.setMissing(i);
			}
		}
		return instance;
	}
	
	public List<Double> generateEntropy(Map<FV, Collection<FV>> fv_list, int numInstances) {
		List<Double> entropies = new ArrayList();
		Iterator<Entry<FV, Collection<FV>>> iterator = fv_list.entrySet().iterator();
		while (iterator.hasNext()) {
			Entry<FV, Collection<FV>> next = iterator.next();
			FV key = next.getKey();
			key.setFrequency((double) next.getValue().size() / numInstances);
			double[] counter = new double[key.getNumLabels()];
			for (FV fv : next.getValue()) {
				int idx = (int) fv.getLabel();
				counter[idx]++;
			}
			key.setEntropy(calculateEntropy(counter, next.getValue().size()));
			entropies.add(key.getEntropy());
		}
		return entropies;
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
	
	public double thresholdSelection(double threshold, Double[] values, ThresholdType thr_alg)
	{
		Double mean = MathHelper.getInstance().calculateAverage(values);
		Double stdev = MathHelper.getInstance().calculateStdev(mean, values);
		Double[] q = MathHelper.getInstance().calculateQuartile(values);
		// System.out.println("Mean: " + mean);
		// System.out.println("Stdev: " + stdev);
		// System.out.println("Mean + Stdev: " + (mean + stdev));
		// System.out.println("Mean - Stdev: " + (mean - stdev));
		// System.out.println("Q1: " + q[0]);
		// System.out.println("Q2: " + q[1]);
		// System.out.println("Q3: " + q[2]);
		// threshold = mean + stdev; // Force using specified threshold
		switch (thr_alg) {
		case Mean:
			threshold = mean;
			break;
		case MeanMin:
			threshold = mean - stdev;
			break;
		case MeanPlus:
			threshold = mean + stdev;
			break;
		case Q1:
			threshold = q[0];
			break;
		case Q2:
			threshold = q[1];
			break;
		case Q3:
			threshold = q[2];
			break;
		default:
			break;
		}
		return threshold;
	}
}
