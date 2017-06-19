package edu.nctu.lalala.fvs.algorithm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import weka.core.Instances;

public class RandomFVS implements IFVS {
	Instances inst;
	Instances output;
	Map<FV, Collection<FV>> fv_list;
	Map<FV, Collection<FV>> filtered_fv;
	int total;
	double epsilon;
	
	public RandomFVS()
	{
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}

	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		this.epsilon = (Double)params[0];
		filtered_fv.putAll(this.fv_list);
	}

	/**
	 * Remove FV randomly, and let some remains
	 * 
	 * @param fv_list
	 * @param total
	 * @return Remaining FVs
	 */
	@Override
	public void applyFVS() {
		// Filter FV based on the algorithm (Random)
		Random random = new Random();
		// Apply removal
		int removed = 0;
		for (FV k : fv_list.keySet()) {
			if (epsilon > random.nextFloat())
			{
				filtered_fv.remove(k);
				removed++;
			}
		}
		System.out.println("Removed: " + removed);
		System.out.println(filtered_fv.size());
	}

	@Override
	public Instances output() {
		Instances output = FVSHelper.getInstance().transformInstances(inst, this.output, filtered_fv);
		return output;
	}
}
