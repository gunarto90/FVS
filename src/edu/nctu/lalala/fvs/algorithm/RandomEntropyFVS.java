package edu.nctu.lalala.fvs.algorithm;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import edu.nctu.lalala.util.MathHelper;
import weka.core.Instances;

public class RandomEntropyFVS implements IFVS {
	Instances inst;
	Instances output;
	Map<FV, Collection<FV>> fv_list;
	Map<FV, Collection<FV>> filtered_fv;
	/**
	 * epsilon is the parameter for removal rate <br/>
	 * 0.0 < epsilon <= 1.0 <br/>
	 * smaller epsilon means higher FV removal probability
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
		preprocessing(inst);
	}

	private void preprocessing(Instances inst) {
		List<Double> entropies = FVSHelper.getInstance().generateEntropy(fv_list, inst.numInstances());
		filtered_fv.putAll(fv_list);

		Map<FV, Double> chi = MathHelper.getInstance().calculateChiSquare(fv_list, this.inst.numClasses());
		double max_phi = 0.0;
		for (Entry<FV, Double> entry : chi.entrySet()) {
			double phi = Math.sqrt(entry.getValue() / this.inst.numInstances()
					/ Math.min(this.inst.numInstances() - 1, this.inst.numAttributes() - 1));
			max_phi = Math.max(phi, max_phi);
			entry.getKey().setPhi(phi);
		}
		if (max_phi > 0.0) {
			for (Entry<FV, Double> entry : chi.entrySet()) {
				double phi = entry.getKey().getPhi() / max_phi;
				entry.getKey().setPhi(phi);
			}
		}
	}

	@Override
	public void applyFVS() {
		Random random = new Random();
		// Apply removal
		int removed = 0;
		for (FV k : fv_list.keySet()) {
			if (k.getEntropy() > (random.nextFloat() * epsilon))
//			if (k.getPhi() < (random.nextFloat() * epsilon)) 
			{
				filtered_fv.remove(k);
				removed++;
			}
		}
		System.out.println("Removed: " + removed);
	}

	@Override
	public Instances output() {
		Instances output = FVSHelper.getInstance().transformInstances(inst, this.output, filtered_fv);
		return output;
	}
}
