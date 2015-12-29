package edu.nctu.lalala.fvs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import edu.nctu.lalala.math.MathHelper;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

enum FVS_Algorithm {
	Original, Random, Threshold, Correlation
}

/**
 * 
 * @author Gunarto Sindoro Njoo
 * @version 1.0
 * @category Preprocessing
 * @see Reference:
 *      https://weka.wikispaces.com/Writing+your+own+Filter+(post+3.5.3)
 */
public class FVS extends Filter {
	FVS_Algorithm algo;
	Double[] params;
	private int numInstances;
	ThresholdType thr_alg = ThresholdType.Iteration;
	// Map<FV, Integer> fv_list = new HashMap<>();

	public FVS(int numInstances, Double... params) {
		algo = FVS_Algorithm.Original;
		this.setNumInstances(numInstances);
		this.params = params;
	}

	public FVS(FVS_Algorithm algo, int numInstances, Double... params) {
		this.algo = algo;
		this.setNumInstances(numInstances);
		this.params = params;
	}

	public FVS(FVS_Algorithm algo, ThresholdType thr_alg, int numInstances, Double... params) {
		this.algo = algo;
		this.thr_alg = thr_alg;
		this.setNumInstances(numInstances);
		this.params = params;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -627598689595987795L;

	public String globalInfo() {
		return "Feature value selection, a preprocessing framework for Data Mining communities.";
	}

	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.enableAllAttributes();
		result.enableAllClasses();
		// filter doesn't need class to be set
		result.enable(Capability.NO_CLASS);
		return result;
	}

	/**
	 * Feature selection process is handled here ... <br/>
	 * Doing feature value selection by using specified algorithm (Threshold or
	 * Correlation)
	 */
	public boolean batchFinished() throws Exception {
		if (getInputFormat() == null)
			throw new NullPointerException("No input instance format defined");

		// output format still needs to be set (depends on first batch of data)
		if (!isFirstBatchDone()) {
			Instances outFormat = new Instances(getInputFormat(), 0);
			setOutputFormat(outFormat);
		}

		Multimap<FV, FV> fv_list = ArrayListMultimap.create();
		Instances inst = getInputFormat();
		Double[] average = calculateAverage(inst);
		// double[] stdev = calculateStdev(inst, average);
		int numOfClassLabels = inst.numClasses();
		int numCols = 0;
		// Instances outFormat = getOutputFormat();
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance ins = inst.instance(i);
			// Skip the class label
			for (int x = 0; x < ins.numAttributes() - 1; x++) {
				Object value = null;
				numCols = ins.numAttributes();
				try {
					value = ins.stringValue(x);
				} catch (Exception e) {
					value = ins.value(x);
				}
				FV fv = new FV(x, value, ins.classValue());
				fv.setNumLabels(numOfClassLabels);
				if (!fv_list.put(fv, fv)) {
					System.err.println("Couldn't put duplicates: " + fv);
				}
			}
		}
		// Creating Correlation Matrix if using Correlation Algorithm
		Double[][] CM = new Double[numCols][numCols];
		List<Double> corrValues = new ArrayList();
		for (int i = 0; i < numCols; i++) {
			for (int j = 0; j < numCols; j++) {
				CM[i][j] = 0.0;
			}
		}
		// Update CM values
		if (algo == FVS_Algorithm.Correlation) {
			for (int i = 0; i < numCols - 1; i++) {
				for (int j = i + 1; j < numCols - 1; j++) {
					CM[i][j] = calculateLinearCorrelation(inst, average, i, j);
					CM[j][i] = CM[i][j]; // Correlation between x and y is the
											// same with y and x
					if (CM[i][j] != 0)
						corrValues.add(CM[i][j]);
				}
			}
			Collections.sort(corrValues);
			// Print Correlation Matrices
			// for (int i = 0; i < numCols; i++) {
			// for (int j = 0; j < numCols; j++) {
			// System.out.print(String.format("%.3f\t", CM[i][j]));
			// }
			// System.out.println();
			// }
		}
		// Remove possible feature values
		Map<FV, Collection<FV>> original_map = fv_list.asMap();
		// Default parameters
		double percent_filter = 80; // Leave 20% fv left
		double threshold = 0.5;
		double topk = 0.1;
		// Lookup on params
		if (params.length > 0) {
			if (algo == FVS_Algorithm.Random) {
				percent_filter = params[0];
			} else if (algo == FVS_Algorithm.Threshold) {
				threshold = params[0];
			} else if (algo == FVS_Algorithm.Correlation) {
				topk = params[0];
			}
		}

		int total = (int) ((double) original_map.size() * percent_filter);
		Map<FV, Collection<FV>> filtered_map = null;

		// Apply removal based on Algorithm
		switch (this.algo) {
		case Original:
			filtered_map = applyRandomRemoval(original_map, 0);
			break;
		case Random:
			filtered_map = applyRandomRemoval(original_map, total);
			break;
		case Threshold:
			filtered_map = applyThresholdRemoval(original_map, threshold);
			break;
		case Correlation:
			filtered_map = applyCorrelationRemoval(original_map, topk, CM, corrValues);
			break;
		default:
			break;
		}
		// printFVs(fv_list, fv_list.asMap());
		// printFVs(reduced_fv_list, reduced_fv_list.asMap());
		// double avg_o_en = calculateAverageEntropy(original_map);
		// double avg_f_en = calculateAverageEntropy(filtered_map);

		// System.out.println("Number of feature value (Original)\tNumber of
		// feature value (Filtered)\tAvg Entropy (Original)\tAvg Entropy
		// (Filtered)");
		// System.out.println(String.format("%d\t%d\t%f\t%f",
		// original_map.size(), filtered_map.size(), avg_o_en, avg_f_en));

		// Apply FVS to the instances and push to
		Instances output = applyFVS(inst, filtered_map, average);
		// System.out.println(inst.numInstances());
		// System.out.println(output.numInstances());

		for (int i = 0; i < output.numInstances(); i++) {
			// System.out.println(output.instance(i));
			push(output.instance(i));
		}

		original_map.clear();
		filtered_map.clear();
		fv_list.clear();

		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;

		return (numPendingOutput() != 0);
	}

	private List<Double> getListEntropy(final Map<FV, Collection<FV>> map) {
		List<Double> result = new ArrayList<Double>();
		for (FV fv : map.keySet()) {
			result.add(fv.getEntropy());
		}
		return result;
	}

	private double calculateAverageEntropy(final Map<FV, Collection<FV>> map) {
		double result = 0.0;
		int count = 0;
		for (FV fv : map.keySet()) {
			result += fv.getEntropy();
			count++;
		}
		if (count == 0)
			return result;
		return result / count;
	}

	/**
	 * Calculate average of every columns
	 * 
	 * @param inst
	 * @return
	 */
	private Double[] calculateAverage(Instances inst) {
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
	private Double calculateLinearCorrelation(Instances inst, Double[] average, int x, int y) {
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

	private void printFVs(Multimap<FV, FV> reduced_fv_list, Map<FV, Collection<FV>> map) {
		for (FV key : map.keySet()) {
			System.out.println(key + "--" + "\t" + reduced_fv_list.get(key).size());
		}
	}

	private Instances applyFVS(Instances inst, Map<FV, Collection<FV>> map, Double[] substitution) {
		Instances output = getOutputFormat();
		Set<FV> set = map.keySet();
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

	private Instance getFVSFilteredInstance(Instances output, Instance old_inst, List<List<Value>> list,
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

	protected Instances determineOutputFormat(Instances inputFormat) {
		Instances result = new Instances(inputFormat, 0);
		// result.insertAttributeAt(new Attribute("bla"),
		// result.numAttributes());
		return result;
	}

	/**
	 * *************************************************************************
	 * Write FVS Algorithm here
	 * *************************************************************************
	 **/

	/**
	 * Remove FV randomly, and let some remains
	 * 
	 * @param fv_list
	 * @param total
	 * @return Remaining FVs
	 */
	public Map<FV, Collection<FV>> applyRandomRemoval(Map<FV, Collection<FV>> fv_list, int total) {
		if (total == 0)
			return fv_list;
		Map<FV, Collection<FV>> result = new HashMap<FV, Collection<FV>>();
		result.putAll(fv_list);
		Random r = new Random();
		// Transfer from map to list first
		List<FV> keys = new ArrayList<>();
		for (FV k : fv_list.keySet())
			keys.add(k);
		// int total = r.nextInt(keys.size()); // Randomly remove number of
		// items
		while (total > 0 && keys.size() > 0) {
			// System.out.println(keys.size());
			int index = r.nextInt(keys.size());
			result.remove(keys.get(index));
			keys.remove(index);
			total--;
		}
		return result;
	}

	/**
	 * Remove FV based on its threshold, if it is over <b> threshold </b> then
	 * it would be removed. Let the FVs that has entropy below threshold
	 * remains.
	 * 
	 * @param fv_list
	 * @param threshold
	 * @return Remaining FVs
	 */
	public Map<FV, Collection<FV>> applyThresholdRemoval(Map<FV, Collection<FV>> fv_list, double threshold) {
		Map<FV, Collection<FV>> result = new HashMap<FV, Collection<FV>>();
		List<Double> entropies = generateEntropy(fv_list);
		result.putAll(fv_list);
		// See mean, Q1~Q3 values for entropy threshold
		Double[] temp = new Double[entropies.size()];
		temp = entropies.toArray(temp);
		Double mean = MathHelper.getInstance().calculateAverage(temp);
		Double stdev = MathHelper.getInstance().calculateStdev(mean, temp);
		Double[] q = MathHelper.getInstance().calculateQuartile(temp);
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
		// Apply removal
		for (FV k : fv_list.keySet()) {
			if (k.getEntropy() > threshold)
				result.remove(k);
		}
		return result;
	}

	// Instead of top-k, we select top-k percents data
	public Map<FV, Collection<FV>> applyCorrelationRemoval(Map<FV, Collection<FV>> fv_list, double topk, Double[][] CM,
			List<Double> corrValues) {
		Map<FV, Collection<FV>> result = new HashMap<FV, Collection<FV>>();
		generateEntropy(fv_list);
		Double mean;
		Double stdev;
		Double[] q = new Double[3]; // Q1, Q2, Q3
		Double[] temp = new Double[corrValues.size()];
		temp = corrValues.toArray(temp);
		// Calculate mean, Q1~Q3
		mean = MathHelper.getInstance().calculateAverage(temp);
		stdev = MathHelper.getInstance().calculateStdev(mean, temp);
		q = MathHelper.getInstance().calculateQuartile(temp);
		// System.out.println("Mean: " + mean);
		// System.out.println("Stdev: " + stdev);
		// System.out.println("Mean + Stdev: " + (mean + stdev));
		// System.out.println("Mean - Stdev: " + (mean - stdev));
		// System.out.println("Q1: " + q[0]);
		// System.out.println("Q2: " + q[1]);
		// System.out.println("Q3: " + q[2]);
		// Selecting correlation threshold
		double corrThreshold = q[1]; // Use Q3 / Q2 / Mean+Stdev as corr
										// threshold
		switch (thr_alg) {
		case Mean:
			corrThreshold = mean;
			break;
		case MeanMin:
			corrThreshold = mean - stdev;
			break;
		case MeanPlus:
			corrThreshold = mean + stdev;
			break;
		case Q1:
			corrThreshold = q[0];
			break;
		case Q2:
			corrThreshold = q[1];
			break;
		case Q3:
			corrThreshold = q[2];
			break;
		default:
			break;
		}
		/**
		 * Create several list for each correlated features
		 */
		List<List<FV>> correlatedFV = new ArrayList<>();
		Set<FV> fvs = fv_list.keySet();
		int[] selectedColumns = new int[CM.length];
		for (int i = 0; i < CM.length; i++) {
			for (int j = i + 1; j < CM.length; j++) {
				/**
				 * For each correlated feature, put all FV into list and do
				 * selection based on top-k percent
				 */
				if (CM[i][j] >= corrThreshold) {
					// Mark selectedColumns
					selectedColumns[i] = 1;
					selectedColumns[j] = 0;
					// Add into list
					List<FV> list = new ArrayList<>();
					for (FV fv : fvs) {
						if (fv.getFeature() == i || fv.getFeature() == j) {
							list.add(fv);
						}
					}
					selectSubsetTopKPercent(topk, correlatedFV, list);
				}
			}
		}
		/**
		 * Add remaining columns which are not selected (not correlated)
		 */
		for (int i = 0; i < selectedColumns.length; i++) {
			if (selectedColumns[i] == 0) {
				List<FV> list = new ArrayList<>();
				for (FV fv : fvs) {
					if (fv.getFeature() == i) {
						list.add(fv);
					}
				}
				selectSubsetTopKPercent(topk, correlatedFV, list);
			}
		}
		/**
		 * Merge all correlated FV set and non-correlated FV set
		 */
		for (List<FV> list : correlatedFV) {
			for (FV fv : list) {
				result.put(fv, null);
			}
		}
		/*
		 * TODO Correlation FVS 1. Create several list for each correlated
		 * features 2. For each correlated feature, put all FV into list and do
		 * selection based on top-k percent 3. When doing selection, check on
		 * every list (because every correlated feature would have different
		 * list and non-correlated feature, would have a separate list). For
		 * example: there is 5 features A, B, C, D, and E. Then, correlated
		 * pairs are: A with B, A with C, A with D, B with D, and C with D. Then
		 * we would have several list for them: AB list, AC list, AD list, BD
		 * list, CD list, and E list. (E is a non-correlated feature, thus have
		 * a separate list by itself). 4. Evaluate the parameters: top-k
		 * percents and threshold selection (for correlation).
		 */
		return result;
	}

	private void selectSubsetTopKPercent(double topk, List<List<FV>> correlatedFV, List<FV> list) {
		Collections.sort(list);
		int limit = (int) Math.min((int) (list.size() * topk) + 1, list.size() - 1); // Prevent
																						// out-of-size
		limit = (int) Math.max(limit, 0); // Prevent negative
		list = list.subList(0, limit);
		correlatedFV.add(list);
	}

	private List<Double> generateEntropy(Map<FV, Collection<FV>> fv_list) {
		List<Double> entropies = new ArrayList();
		Iterator<Entry<FV, Collection<FV>>> iterator = fv_list.entrySet().iterator();
		while (iterator.hasNext()) {
			Entry<FV, Collection<FV>> next = iterator.next();
			FV key = next.getKey();
			key.setFrequency((double) next.getValue().size() / this.numInstances);
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

	/**
	 * *************************************************************************
	 * Utility function here
	 * *************************************************************************
	 **/

	private double calculateEntropy(double[] counter, double frequency) {
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

	int getNumInstances() {
		return numInstances;
	}

	void setNumInstances(int numInstances) {
		this.numInstances = numInstances;
	}
}
