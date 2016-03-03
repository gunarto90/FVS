package edu.nctu.lalala.fvs;

import edu.nctu.lalala.enums.FVS_Algorithm;
import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.algorithm.CorrelationFVS;
import edu.nctu.lalala.fvs.algorithm.EntropyFVS;
import edu.nctu.lalala.fvs.algorithm.RandomFVS;
import edu.nctu.lalala.fvs.interfaces.IFVS;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instances;
import weka.filters.Filter;


/**
 * 
 * @author Gunarto Sindoro Njoo
 * @version 1.0
 * @category Preprocessing
 * @see Reference:
 *      https://weka.wikispaces.com/Writing+your+own+Filter+(post+3.5.3)
 */
public class FVS_Filter extends Filter {
	FVS_Algorithm algo;
	Double[] params;
	private int numInstances;
	ThresholdType thr_alg = ThresholdType.Iteration;

	public FVS_Filter(int numInstances, Double... params) {
		algo = FVS_Algorithm.Original;
		this.setNumInstances(numInstances);
		this.params = params;
	}

	public FVS_Filter(FVS_Algorithm algo, int numInstances, Double... params) {
		this.algo = algo;
		this.setNumInstances(numInstances);
		this.params = params;
	}

	public FVS_Filter(FVS_Algorithm algo, ThresholdType thr_alg, int numInstances, Double... params) {
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

		// Initialization
		Instances inst = getInputFormat();
		Instances output = getOutputFormat();
		IFVS fvs = null;
		
		// Apply removal based on Algorithm
		switch (this.algo) {
		case Original:
			fvs = new RandomFVS();
			fvs.input(inst, output, (Double)(0.0));
			break;
		case Random:
			fvs = new RandomFVS();
			double percent_filter = 80.0;
			if (params.length > 0) percent_filter = params[0];
			fvs.input(inst, output, percent_filter);
			break;
		case Threshold:
			fvs = new EntropyFVS(thr_alg);
			Double threshold = 0.5;
			if (params.length > 0) threshold = params[0];
			fvs.input(inst, output, threshold);
			break;
		case Correlation:
			fvs = new CorrelationFVS(thr_alg);
			Double topk = 0.1;
			if (params.length > 0) topk = params[0];
			fvs.input(inst, output, topk);
			break;
		default:
			fvs = new RandomFVS();
			fvs.input(inst, output, (Double)(0.0));
			break;
		}
		
		fvs.applyFVS();
		output = fvs.output();

		for (int i = 0; i < output.numInstances(); i++) {
			// System.out.println(output.instance(i));
			push(output.instance(i));
		}

		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;

		return (numPendingOutput() != 0);
	}
	
	/**
	 * *************************************************************************
	 * Utility function here
	 * *************************************************************************
	 **/

	int getNumInstances() {
		return numInstances;
	}

	void setNumInstances(int numInstances) {
		this.numInstances = numInstances;
	}
}
