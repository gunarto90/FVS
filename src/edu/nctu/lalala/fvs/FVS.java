package edu.nctu.lalala.fvs;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
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
	Threshold, Correlation
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
	// Map<FV, Integer> fv_list = new HashMap<>();

	public FVS() {
		algo = FVS_Algorithm.Threshold;
	}

	public FVS(FVS_Algorithm algo) {
		this.algo = algo;
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
	 * Doing feature value selection by using specified algorithm (Threshold or
	 * Correlation)
	 */
	public boolean batchFinished() throws Exception {
		if (getInputFormat() == null)
			throw new NullPointerException("No input instance format defined");

		// output format still needs to be set (depends on first batch of data)
		if (!isFirstBatchDone()) {
			Instances outFormat = new Instances(getInputFormat(), 0);
			// outFormat.insertAttributeAt(new Attribute(
			// "bla-" + getInputFormat().numInstances()),
			// outFormat.numAttributes());
			setOutputFormat(outFormat);
		}

		Multimap<FV, Integer> fv_list = ArrayListMultimap.create();
		Instances inst = getInputFormat();
		Instances outFormat = getOutputFormat();
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
				FV fv = new FV(x, value, ins.classValue());
				if (!fv_list.put(fv, 1)) {
					System.err.println("Couldn't put duplicates: " + fv);
				}
			}
			Instance new_ins = getFilteredInstance(ins, fv_list);
		}
		Multimap<FV, Integer> reduced_fv_list = applyRandomRemoval(fv_list);
		Map<FV, Collection<Integer>> map = reduced_fv_list.asMap();
		for (FV key : map.keySet()) {
			System.out.println(key + "--" + "\t" + reduced_fv_list.get(key).size());
		}
		// Instances inst = getInputFormat();
		// Instances outFormat = getOutputFormat();
		// for (int i = 0; i < inst.numInstances(); i++) {
		// double[] newValues = new double[outFormat.numAttributes()];
		// double[] oldValues = inst.instance(i).toDoubleArray();
		// System.arraycopy(oldValues, 0, newValues, 0, oldValues.length);
		// newValues[newValues.length - 1] = i;
		// push(new Instance(1.0, newValues));
		// }

		flushInput();
		m_NewBatch = true;
		m_FirstBatchDone = true;

		return (numPendingOutput() != 0);
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

	private Multimap<FV, Integer> applyRandomRemoval(Multimap<FV, Integer> fv_list) {
		Multimap<FV, Integer> result = ArrayListMultimap.create();
		result.putAll(fv_list);
		Random r = new Random();
		List<FV> keys = new ArrayList();
		for (FV k : fv_list.keySet())
			keys.add(k);
		int total = r.nextInt(keys.size());
		while (total > 0 && keys.size() > 0) {
//			System.out.println(keys.size());
			int index = r.nextInt(keys.size());
			result.removeAll(keys.get(index));
			keys.remove(index);
			total--;
		}
		return result;
	}

	private Instance getFilteredInstance(Instance ins, Multimap<FV, Integer> fv_list) {
		return new Instance(ins);
	}
}
