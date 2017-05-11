package edu.nctu.lalala.fvs.algorithm;

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import weka.core.Instances;

public class RandomEntropyFVS implements IFVS {
	Instances inst;
	Instances output;
	Map<FV, Collection<FV>> fv_list;
	Map<FV, Collection<FV>> filtered_fv;

	public RandomEntropyFVS() {
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}

	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		preprocessing(inst);
	}

	private void preprocessing(Instances inst) {
		List<Double> entropies = FVSHelper.getInstance().generateEntropy(fv_list, inst.numInstances());
		filtered_fv.putAll(fv_list);
		// See mean, Q1~Q3 values for entropy threshold
		Double[] temp = new Double[entropies.size()];
		temp = entropies.toArray(temp);
	}

	@Override
	public void applyFVS() {
		Random random = new Random();
		// Apply removal
		for (FV k : fv_list.keySet()) {
			if (k.getEntropy() > random.nextFloat())
				filtered_fv.remove(k);
		}
	}

	@Override
	public Instances output() {
		Instances output = FVSHelper.getInstance().transformInstances(inst, this.output, filtered_fv);
		return output;
	}
}
