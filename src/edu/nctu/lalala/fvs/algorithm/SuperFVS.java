package edu.nctu.lalala.fvs.algorithm;

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import weka.core.Instances;

public class SuperFVS implements IFVS {
	protected Instances inst;
	protected Instances output;
	protected Map<FV, Collection<FV>> fv_list;
	protected Map<FV, Collection<FV>> filtered_fv;
	/**
	 * epsilon is the parameter for removal rate <br/>
	 * 0.0 < epsilon <= 1.0 <br/>
	 * larger epsilon means higher FV removal probability
	 */
	protected double epsilon = 1.0;
	
	public SuperFVS()
	{
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}
	
	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		if (params.length > 0)
			this.epsilon = (Double) params[0];
	}

	@Override
	public void applyFVS() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public Instances output() {
		// TODO Auto-generated method stub
		return null;
	}

	public void dumpModel() {
		System.out.println("Dump model");
		if(FVSHelper.getInstance().getDumpModelStatus())
		{
			FVSHelper.getInstance().logFile("Filtered Feature Values");
			FV[] arr = (FV[])filtered_fv.keySet().toArray();
			List<FV> temp = Arrays.asList(arr);
			Collections.sort(temp);
			FVSHelper.getInstance().logFile(temp.toString());
			System.out.println(arr);
		}
	}

}
