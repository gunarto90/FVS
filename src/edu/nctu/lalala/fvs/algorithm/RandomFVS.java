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
	
	public RandomFVS()
	{
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}

	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		this.total = (int) ((double) fv_list.size() * (Double)params[0]);
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
		if (this.total == 0)
			return;
		filtered_fv = new HashMap<FV, Collection<FV>>();
		filtered_fv.putAll(this.fv_list);
		Random r = new Random();
		// Transfer from map to list first
		List<FV> keys = new ArrayList<>();
		for (FV k : this.fv_list.keySet())
			keys.add(k);
		// int total = r.nextInt(keys.size()); // Randomly remove number of
		// items
		int counter = total;
		while (counter > 0 && keys.size() > 0) {
			// System.out.println(keys.size());
			int index = r.nextInt(keys.size());
			filtered_fv.remove(keys.get(index));
			keys.remove(index);
			counter--;
		}
	}

	@Override
	public Instances output() {
		Instances output = FVSHelper.getInstance().transformInstances(inst, this.output, filtered_fv);
		return output;
	}
}
