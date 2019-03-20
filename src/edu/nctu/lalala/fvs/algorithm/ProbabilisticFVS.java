package edu.nctu.lalala.fvs.algorithm;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.Value;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import edu.nctu.lalala.util.MathHelper;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class ProbabilisticFVS implements IFVS {
	Instances inst;
	Instances output;
	Map<FV, Collection<FV>> fv_list;
	Map<FV, Collection<FV>> filtered_fv;
	/**
	 * epsilon is the parameter for removal rate (optimistic removal) <br/>
	 * 0.0 < epsilon <= 1.0 <br/>
	 * higher epsilon means higher FV removal probability
	 */
	double epsilon = 1.0;

	public ProbabilisticFVS() {
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}

	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		if (params.length > 0)
			this.epsilon = (Double) params[0];
		preprocessing(inst);
	}

	private void preprocessing(Instances inst) {
		int[] act_dist = new int[inst.numClasses()];
		for (int i=0; i<inst.numInstances(); i++) {
			act_dist[(int) inst.get(i).classValue()]++;
		}
		double act_ent = MathHelper.getInstance().calculateEntropy(act_dist, inst.numInstances(), inst.numClasses());
//		System.out.println(act_ent);
		
		/* Generating entropy and frequency for each FV */
		FVSHelper.getInstance().generateEntropy(fv_list, inst.numInstances(), inst.numClasses());
		filtered_fv.putAll(fv_list);
		
		double max_ig = 0.0;
		double max_su = 0.0;
		Map<FV, Double> igs = MathHelper.getInstance().calculateIG(filtered_fv, act_dist, act_ent, this.inst.numClasses());
		for (Entry<FV, Double> entry : igs.entrySet()) {
			double ig = entry.getValue();
			double su = 2*ig/(entry.getKey().getEntropy()+act_ent);
			entry.getKey().setIg(ig);
			max_ig = Math.max(max_ig, ig);
			entry.getKey().setSymmetricUncertainty(su);
			max_su = Math.max(max_su, su);
		}
		// Normalize to range 0-1
		if (max_ig > 0.0) {
			for (Entry<FV, Double> entry : igs.entrySet()) {
				double ig = entry.getKey().getIg() / max_ig;
				entry.getKey().setIg(ig);
				if (max_su > 0.0)
				{
					double su = entry.getKey().getSymmetricUncertainty() / max_su;
					entry.getKey().setSymmetricUncertainty(su);
				}
			}
		}

//		Map<FV, Double> chi = MathHelper.getInstance().calculateChi(fv_list, this.inst.numClasses());
//		double max_phi = 0.0;
//		for (Entry<FV, Double> entry : chi.entrySet()) {
//			double phi = Math.sqrt(entry.getValue() / this.inst.numInstances()
//					/ Math.min(this.inst.numInstances() - 1, this.inst.numAttributes() - 1));
//			max_phi = Math.max(phi, max_phi);
//			entry.getKey().setPhi(phi);
//		}
		// Normalize to range 0-1
//		if (max_phi > 0.0) {
//			for (Entry<FV, Double> entry : chi.entrySet()) {
//				double phi = entry.getKey().getPhi() / max_phi;
//				entry.getKey().setPhi(phi);
//			}
//		}
	}

	@Override
	public void applyFVS() {
		
	}

	@Override
	public Instances output() {
		boolean removeInstance = true;
		boolean probabilistic = true;
		boolean average = false;
		Instances output = transformInstances(inst, this.output, filtered_fv, removeInstance, probabilistic, average);
		writeToFile(output, true);
		return output;
	}
	
	private Instances transformInstances(Instances inst, Instances output, Map<FV, Collection<FV>> map,
			boolean removeInstance, boolean probabilistic, boolean average) {
		Set<FV> set = map.keySet();
		Double[] substitution = FVSHelper.getInstance().calculateAverage(inst);
		// Prepare the list
		// First level indicate which attribute the FVS resides in
		List<Map<Value, FV>> list = new ArrayList<>();
		for (int i = 0; i < inst.numAttributes(); i++) {
			list.add(new HashMap<Value, FV>());
		}
		// Build the data structure
		for (FV fv : set) {
			Value v = new Value(fv.getValue().toString());
			list.get(fv.getFeature()).put(v, fv);
		}
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance instance = getFVSFilteredInstance(output, inst.instance(i), list, substitution, removeInstance, probabilistic, average);
			if (removeInstance && instance == null)
				continue;
			output.add(instance);
		}
		if (FVSHelper.getInstance().getDebugStatus()) {
			System.out.println("Input: " + inst.numInstances());
			System.out.println("Output: " + output.numInstances());
		}
		return output;
	}
	
	private Instance getFVSFilteredInstance(Instances output, Instance old_inst, List<Map<Value, FV>> map,
			Double[] substitution, boolean removeInstance, boolean probabilistic, boolean average) {
		double[] oldValues = old_inst.toDoubleArray();
		Instance instance = new DenseInstance(old_inst);
		int count_miss = 0;
		Random random = new Random();
		for (int i = 0; i < oldValues.length - 1; i++) {
			Value v = new Value(oldValues[i]);
			FV fv = map.get(i).getOrDefault(v, null);
			if (fv == null) {
				FVSHelper.getInstance().replaceValue(substitution, average, instance, i);
				count_miss++;
			} else if (old_inst.isMissing(i)) {
				count_miss++;
			} else if (fv != null && probabilistic) {
				double rr = random.nextFloat() * epsilon;
				boolean condition = fv.getIg() < rr;
				if (FVSHelper.getInstance().getInformationMetric().equals("entropy"))
					condition = fv.getEntropy() > rr;
				else
					condition = fv.getIg() < rr;
				if (condition) 
				{
					FVSHelper.getInstance().replaceValue(substitution, average, instance, i);
					count_miss++;
				}
			}
		}
		if (removeInstance) {
			/* Remove the instance using miss rate probability */
			double miss_rate = (double) count_miss / oldValues.length;
			if (miss_rate > random.nextFloat()) {
				instance = null;
			}
		}
		return instance;
	}
	
	private void writeToFile(Instances output, boolean filtered) {
		if (FVSHelper.getInstance().getDumpModelStatus()) {
			ArffSaver saver = new ArffSaver();
			saver.setInstances(output);
			try {
				File f = null;
				if (filtered)
					f = File.createTempFile(String.format("Random %.1f Probabilistic FVS", this.epsilon), ".arff");
				else
					f = File.createTempFile(String.format("Original %.1f Random Probabilistic FVS", this.epsilon), ".arff");
				System.out.println(f.getAbsolutePath());
				saver.setFile(f);
				saver.writeBatch();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}