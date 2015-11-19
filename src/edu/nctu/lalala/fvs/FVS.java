package edu.nctu.lalala.fvs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

enum FVS_Algorithm {
	Null, Random, Threshold, Correlation
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
	// Map<FV, Integer> fv_list = new HashMap<>();

	public FVS(int numInstances, Double... params) {
		algo = FVS_Algorithm.Null;
		this.setNumInstances(numInstances);
		this.params = params;
	}

	public FVS(FVS_Algorithm algo, int numInstances, Double... params) {
		this.algo = algo;
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
		double[] substitution = calculateAverage(inst);
		int numOfClassLabels = inst.numClasses();
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
				// TODO Class label using String
				// Class value as a double
				// How about String?
				// Not implemented yet
				FV fv = new FV(x, value, ins.classValue());
				fv.setNumLabels(numOfClassLabels);
				if (!fv_list.put(fv, fv)) {
					System.err.println("Couldn't put duplicates: " + fv);
				}
			}
		}
		// Remove possible feature values
		Map<FV, Collection<FV>> original_map = fv_list.asMap();
		// Default parameters
		double percent_filter = 80; // Leave 20% fv left
		double threshold = 0.5;
		// Lookup on params
		if(params.length>0)
		{
			if(algo==FVS_Algorithm.Random)
			{
				percent_filter = params[0];
			}
			else if(algo==FVS_Algorithm.Threshold)
			{
				threshold = params[0];
			}
		}
		
		int total = (int)((double)original_map.size() * percent_filter);
		Map<FV, Collection<FV>> filtered_map = null;

		// Apply removal based on Algorithm
		switch (this.algo) {
		case Null:
			filtered_map = applyRandomRemoval(original_map, 0);
			break;
		case Random:
			filtered_map = applyRandomRemoval(original_map, total);
			break;
		case Threshold:
			filtered_map = applyThresholdRemoval(original_map, threshold);
			break;
		case Correlation:
			filtered_map = applyCorrelationRemoval(original_map);
			break;
		default:
			break;
		}
		// printFVs(fv_list, fv_list.asMap());
		// printFVs(reduced_fv_list, reduced_fv_list.asMap());
//		double avg_o_en = calculateAverageEntropy(original_map);
//		double avg_f_en = calculateAverageEntropy(filtered_map);
		
		//System.out.println("Number of feature value (Original)\tNumber of feature value (Filtered)\tAvg Entropy (Original)\tAvg Entropy (Filtered)");
		//System.out.println(String.format("%d\t%d\t%f\t%f", original_map.size(), filtered_map.size(), avg_o_en, avg_f_en));

		// Apply FVS to the instances and push to
		// TODO Comment this if not debug, or give IS_DEBUG options
		Instances output = applyFVS(inst, filtered_map, substitution);
		// System.out.println(inst.numInstances());
		// System.out.println(output.numInstances());

		for (int i = 0; i < output.numInstances(); i++) {
//			System.out.println(output.instance(i));
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
	
	private List<Double> getListEntropy(final Map<FV, Collection<FV>> map)
	{
		List<Double> result = new ArrayList<Double>();
		for(FV fv : map.keySet())
		{
			result.add(fv.getEntropy());
		}
		return result;
	}
	
	private double calculateAverageEntropy(final Map<FV, Collection<FV>> map)
	{
		double result = 0.0;
		int count = 0;
		for(FV fv : map.keySet())
		{
			result += fv.getEntropy();
			count++;
		}
		if(count == 0) return result;
		return result/count;
	}

	private double[] calculateAverage(Instances inst) {
		double[] substitution = new double[inst.numAttributes() - 1];
		for (int i = 0; i < inst.numInstances(); i++) {
			for (int x = 0; x < inst.instance(i).numAttributes() - 1; x++) {
				substitution[x] += inst.instance(i).value(x);
			}
		}
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			substitution[i] /= inst.numInstances();
		}
		return substitution;
	}

	private void printFVs(Multimap<FV, FV> reduced_fv_list, Map<FV, Collection<FV>> map) {
		for (FV key : map.keySet()) {
			System.out.println(key + "--" + "\t" + reduced_fv_list.get(key).size());
		}
	}

	private Instances applyFVS(Instances inst, Map<FV, Collection<FV>> map, double[] substitution) {
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
			double[] substitution) {
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

	// TODO Feature-Value Selection logic HERE
	/**
	 * Doing feature value selection by using specified algorithm (Threshold or
	 * Correlation)
	 * 
	 * @param inst
	 * @return
	 */
	// protected Instances process(Instances inst) {
	// Instances result = new Instances(inst, 0);
	// for (int i = 0; i < inst.numInstances(); i++) {
	// Instance ins = inst.instance(i);
	// for (int x = 0; x < ins.numAttributes(); x++) {
	// Object value = null;
	// try {
	// value = ins.stringValue(x);
	// } catch (Exception e) {
	// value = ins.value(x);
	// }
	// FV fv = new FV(x, value);
	// fv_list.put(fv, 1);
	// }
	// }
	// for (Iterator<Entry<FV, Integer>> iterator =
	// fv_list.entries().iterator(); iterator.hasNext();) {
	// Entry<FV, Integer> entry = iterator.next();
	// System.out.println(entry.getKey() + ":" + entry.getValue());
	// }
	//
	// return result;
	// }

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
		Iterator<Entry<FV, Collection<FV>>> iterator = fv_list.entrySet().iterator();
		// Calculate entropy and frequency for each
		while(iterator.hasNext())
		{
			Entry<FV, Collection<FV>> next = iterator.next();
			FV key = next.getKey();
			key.setFrequency((double)next.getValue().size()/this.numInstances);
			double[] counter = new double[key.getNumLabels()];
			for(FV fv: next.getValue())
			{
				int idx = (int) fv.getLabel();
				counter[idx]++;
			}
			key.setEntropy(calculateEntropy(counter, next.getValue().size()));
		}
		result.putAll(fv_list);
		// Apply removal
		for (FV k : fv_list.keySet()) {
			if (k.getEntropy() > threshold)
				result.remove(k);
		}
		return result;
	}

	public Map<FV, Collection<FV>> applyCorrelationRemoval(Map<FV, Collection<FV>> fv_list) {
		Map<FV, Collection<FV>> result = new HashMap<FV, Collection<FV>>();
		result.putAll(fv_list);
		return result;
	}
	
	/**
	 * *************************************************************************
	 * Utility function here
	 * *************************************************************************
	 **/
	
	private double calculateEntropy(double[] counter, double frequency)
	{
		if(frequency==0) return 0;
		double entropy = 0;
		double[] p = new double[counter.length];
		for(int i=0; i<counter.length; i++)
		{
			p[i] = counter[i]/frequency;
		}
		for(int i=0; i<p.length; i++)
		{
			if(p[i]==0.0F) continue;
			entropy -= p[i] * (Math.log(p[i])/Math.log(p.length));
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
