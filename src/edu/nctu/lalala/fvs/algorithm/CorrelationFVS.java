package edu.nctu.lalala.fvs.algorithm;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.CorrelationMatrix;
import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import edu.nctu.lalala.util.FVSHelper;
import weka.core.Instances;

public class CorrelationFVS implements IFVS {
	Instances inst;
	Instances output;
	Map<FV, Collection<FV>> fv_list;
	Map<FV, Collection<FV>> filtered_fv;
	ThresholdType thr_alg;
	double topk;
	double corrThreshold;
	Double[][] CM;
	List<Double> corrValues;

	public CorrelationFVS(ThresholdType thr_alg) {
		this.thr_alg = thr_alg;
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
	}
	@Override
	public void input(Instances inst, Instances output, Object... params) {
		this.inst = inst;
		this.output = output;
		this.fv_list = FVSHelper.getInstance().extractValuesFromData(inst);
		this.topk = (Double) params[0];
		preprocessing(inst);
	}
	private void preprocessing(Instances inst) {
		CorrelationMatrix cm = FVSHelper.getInstance().generateCorrelationMatrix(inst);
		this.CM = cm.getCM();
		this.corrValues = cm.getCorrValues();
		Double[] temp = new Double[corrValues.size()];
		temp = corrValues.toArray(temp);
		this.corrThreshold = FVSHelper.getInstance().thresholdSelection(0, temp, this.thr_alg);
		FVSHelper.getInstance().generateEntropy(fv_list, inst.numInstances());
	}

	@Override
	public void applyFVS() {
		this.filtered_fv = new HashMap<FV, Collection<FV>>();
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
				filtered_fv.put(fv, null);
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
	}

	@Override
	public Instances output() {
		Instances output = FVSHelper.getInstance().transformInstances(inst, this.output, filtered_fv);
		return output;
	}

	private void selectSubsetTopKPercent(double topk, List<List<FV>> correlatedFV, List<FV> list) {
		Collections.sort(list);
		int limit = (int) Math.min((int) (list.size() * topk) + 1, list.size() - 1); // Prevent
																						// out-of-size
		limit = (int) Math.max(limit, 0); // Prevent negative
		list = list.subList(0, limit);
		correlatedFV.add(list);
	}
}
