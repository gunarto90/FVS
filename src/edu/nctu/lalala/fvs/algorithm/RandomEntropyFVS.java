package edu.nctu.lalala.fvs.algorithm;

import java.io.File;
import java.io.IOException;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import edu.nctu.lalala.util.MathHelper;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class RandomEntropyFVS implements IFVS {
	Instances inst;
	Instances output;
	Map<FV, Collection<FV>> fv_list;
	Map<FV, Collection<FV>> filtered_fv;
	/**
	 * epsilon is the parameter for removal rate <br/>
	 * 0.0 < epsilon <= 1.0 <br/>
	 * larger epsilon means higher FV removal probability
	 */
	double epsilon = 1.0;

	public RandomEntropyFVS() {
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}

	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		if (params.length > 0)
			this.epsilon = (Double) params[0];
//		writeToFile(inst, false);
		preprocessing(inst);
	}

	private void preprocessing(Instances inst) {
		int[] act_dist = new int[inst.numClasses()];
		for (int i = 0; i < inst.numInstances(); i++) {
			act_dist[(int) inst.get(i).classValue()]++;
		}
		double act_ent = MathHelper.getInstance().calculateEntropy(act_dist, inst.numInstances(), inst.numClasses());
		// System.out.println(act_ent);

		/* Generating entropy and frequency for each FV */
		FVSHelper.getInstance().generateEntropy(fv_list, inst.numInstances(), inst.numClasses());
		filtered_fv.putAll(fv_list);

		double max_ig = 0.0;
		double max_su = 0.0;
		Map<FV, Double> igs = MathHelper.getInstance().calculateIG(fv_list, act_dist, act_ent, this.inst.numClasses());
		for (Entry<FV, Double> entry : igs.entrySet()) {
			double ig = entry.getValue();
			double su = 2 * ig / (entry.getKey().getEntropy() + act_ent);
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
				if (max_su > 0.0) {
					double su = entry.getKey().getSymmetricUncertainty() / max_su;
					entry.getKey().setSymmetricUncertainty(su);
				}
			}
		}

		// Map<FV, Double> chi = MathHelper.getInstance().calculateChi(fv_list,
		// this.inst.numClasses());
		// double max_phi = 0.0;
		// for (Entry<FV, Double> entry : chi.entrySet()) {
		// double phi = Math.sqrt(entry.getValue() / this.inst.numInstances()
		// / Math.min(this.inst.numInstances() - 1, this.inst.numAttributes() -
		// 1));
		// max_phi = Math.max(phi, max_phi);
		// entry.getKey().setPhi(phi);
		// }
		// Normalize to range 0-1
		// if (max_phi > 0.0) {
		// for (Entry<FV, Double> entry : chi.entrySet()) {
		// double phi = entry.getKey().getPhi() / max_phi;
		// entry.getKey().setPhi(phi);
		// }
		// }
	}

	@Override
	public void applyFVS() {
		boolean show_information_stats = false;
		Random random = new Random();
		// Apply removal
		int removed = 0;
		double average_entropy = 0.0;
		double average_ig = 0.0;
		for (FV k : fv_list.keySet()) {
			if (show_information_stats) {
				average_entropy += k.getEntropy();
				average_ig += k.getIg();
			}
			double rr = random.nextFloat() * epsilon;
			boolean condition = k.getIg() < rr;
			if (FVSHelper.getInstance().getInformationMetric().equals("entropy"))
				condition = k.getEntropy() > rr;
			else
				condition = k.getIg() < rr;
			// if (k.getEntropy() > (random.nextFloat() * epsilon))
			// if (k.getPhi() < (random.nextFloat() * epsilon))
			// if (k.getIg() < (random.nextFloat() * epsilon))
			// if(k.getSymmetricUncertainty() < (random.nextFloat() * epsilon))
			if (condition) {
				filtered_fv.remove(k);
				removed++;
			}
		}
		if (show_information_stats) {
			average_entropy /= fv_list.size();
			average_ig /= fv_list.size();
			System.out.println("Entropy: " + average_entropy);
			System.out.println("IG: " + average_ig);
		}
		if (FVSHelper.getInstance().getDebugStatus()) {
			System.out.println("Removed: " + removed);
			System.out.println("Filtered FV Size: " + filtered_fv.size());
		}
	}

	@Override
	public Instances output() {
		boolean removeInstance = true;
		Instances output = FVSHelper.getInstance().transformInstances(inst, this.output, filtered_fv, removeInstance);
//		writeToFile(output, true);
		return output;
	}

	private void writeToFile(Instances output, boolean filtered) {
		if (FVSHelper.getInstance().getDumpModelStatus()) {
			ArffSaver saver = new ArffSaver();
			saver.setInstances(output);
			try {
				File f = null;
				if (filtered)
					f = File.createTempFile(String.format("Random %.1f Entropy FVS", this.epsilon), ".arff");
				else
					f = File.createTempFile(String.format("Original %.1f Random Entropy FVS", this.epsilon), ".arff");
				System.out.println(f.getAbsolutePath());
				saver.setFile(f);
				saver.writeBatch();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
}
