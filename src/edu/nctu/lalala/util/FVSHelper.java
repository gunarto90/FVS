package edu.nctu.lalala.util;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import edu.nctu.lalala.enums.ClassifierType;
import edu.nctu.lalala.enums.DiscretizationType;
import edu.nctu.lalala.enums.Preprocessing_Algorithm;
import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.CorrelationMatrix;
import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.Value;
import edu.nctu.lalala.main.Main;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

public class FVSHelper {
	private static final long timestamp = new Date().getTime();
	private static final String CONFIG_FOLDER = "config" + "/";
	private static final String LOG_FOLDER = "log" + "/";
	private String INTERMEDIATE_FOLDER = "intermediate" + "/";

	private FVSHelper() {
		System.err.println(timestamp);
	}

	private static final FVSHelper singleton = new FVSHelper();

	public static FVSHelper getInstance() {
		return singleton;
	}

	public Map<FV, Collection<FV>> extractValuesFromData(Instances inst) {
		Multimap<FV, FV> fv_list = ArrayListMultimap.create();
		// Instances outFormat = getOutputFormat();
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
				fv.setNumLabels(inst.numClasses());
				if (!fv_list.put(fv, fv)) {
					System.err.println("Couldn't put duplicates: " + fv);
				}
			}
		}
		Map<FV, Collection<FV>> original_map = fv_list.asMap();
		return original_map;
	}

	public CorrelationMatrix generateCorrelationMatrix(Instances inst) {
		CorrelationMatrix result = new CorrelationMatrix();
		Double[][] CM = new Double[inst.numAttributes()][inst.numAttributes()];
		Double[] average = calculateAverage(inst);
		for (int i = 0; i < inst.numAttributes(); i++) {
			for (int j = 0; j < inst.numAttributes(); j++) {
				CM[i][j] = 0.0;
			}
		}
		// Update CM values
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			for (int j = i + 1; j < inst.numAttributes() - 1; j++) {
				CM[i][j] = calculateLinearCorrelation(inst, average, i, j);
				CM[j][i] = CM[i][j]; // Correlation between x and y is the
										// same with y and x
				if (CM[i][j] != 0)
					result.getCorrValues().add(CM[i][j]);
			}
		}
		Collections.sort(result.getCorrValues());
		result.setCM(CM);
		return result;
	}

	/**
	 * Calculate average of every columns
	 * 
	 * @param inst
	 * @return
	 */
	public Double[] calculateAverage(Instances inst) {
		Double[] average = new Double[inst.numAttributes() - 1];
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			average[i] = 0.0;
		}
		for (int i = 0; i < inst.numInstances(); i++) {
			for (int x = 0; x < inst.instance(i).numAttributes() - 1; x++) {
				Instance ins = inst.instance(i);
				if (ins != null && !Double.isNaN(ins.value(x)))
					average[x] += ins.value(x);
			}
		}
		for (int i = 0; i < inst.numAttributes() - 1; i++) {
			average[i] /= inst.numInstances();
		}
		return average;
	}

	/**
	 * Calculate linear correlation between 2 columns Reference:
	 * https://en.wikipedia.org/wiki/Pearson_product-
	 * moment_correlation_coefficient
	 * 
	 * @param inst
	 * @param average
	 * @param x
	 *            Column 1
	 * @param y
	 *            Column 2
	 * @return
	 */
	public Double calculateLinearCorrelation(Instances inst, Double[] average, int x, int y) {
		double corr = 0;
		double top, rootXiXbar, rootYiYbar, bot;
		top = rootXiXbar = rootYiYbar = 0;
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance ins = inst.instance(i);
			if (ins != null && !Double.isNaN(ins.value(x)) && !Double.isNaN(ins.value(y))) {
				top += (ins.value(x) - average[x]) * (ins.value(y) - average[y]);
				rootXiXbar += Math.pow(ins.value(x) - average[x], 2);
				rootYiYbar += Math.pow(ins.value(y) - average[y], 2);
			}
		}
		rootXiXbar = Math.sqrt(rootXiXbar);
		rootYiYbar = Math.sqrt(rootYiYbar);
		bot = rootXiXbar * rootYiYbar;
		if (bot != 0) {
			corr = top / bot;
		}
		return corr;
	}

	public Instances transformInstances(Instances inst, Instances output, Map<FV, Collection<FV>> map) {
		Set<FV> set = map.keySet();
		Double[] substitution = calculateAverage(inst);
		// Prepare the list
		// First level indicate which attribute the FVS resides in
		List<List<Value>> list = new ArrayList<>();
		for (int i = 0; i < inst.numAttributes(); i++) {
			list.add(new ArrayList<Value>());
		}
		// Build the data structure
		for (FV fv : set) {
			list.get(fv.getFeature()).add(new Value(fv.getValue().toString()));
		}
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance instance = getFVSFilteredInstance(output, inst.instance(i), list, substitution);
			output.add(instance);
		}
		return output;
	}

	public Instance getFVSFilteredInstance(Instances output, Instance old_inst, List<List<Value>> list,
			Double[] substitution) {
		double[] oldValues = old_inst.toDoubleArray();
		Instance instance = new DenseInstance(old_inst);
		// Change with value that is available
		for (int i = 0; i < oldValues.length - 1; i++) {
			// System.out.println(oldValues[i]);
			// System.out.println(list.get(i));
			// System.out.println("############################");
			// If list doesn't contain, then delete
			Value v = new Value(oldValues[i]);
			int idx = list.get(i).indexOf(v);
			// If not found in the index
			if (idx == -1) {
				// Change with substitution
				instance.setValue(i, substitution[i]);
				// Change into missing
				// instance.setMissing(i);
			}
		}
		return instance;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public List<Double> generateEntropy(Map<FV, Collection<FV>> fv_list, int numInstances) {
		List<Double> entropies = new ArrayList();
		Iterator<Entry<FV, Collection<FV>>> iterator = fv_list.entrySet().iterator();
		while (iterator.hasNext()) {
			Entry<FV, Collection<FV>> next = iterator.next();
			FV key = next.getKey();
			key.setFrequency((double) next.getValue().size() / numInstances);
			double[] counter = new double[key.getNumLabels()];
			for (FV fv : next.getValue()) {
				int idx = (int) fv.getLabel();
				counter[idx]++;
			}
			key.setEntropy(calculateEntropy(counter, next.getValue().size()));
			entropies.add(key.getEntropy());
		}
		return entropies;
	}

	public double calculateEntropy(double[] counter, double frequency) {
		if (frequency == 0)
			return 0;
		double entropy = 0;
		double[] p = new double[counter.length];
		for (int i = 0; i < counter.length; i++) {
			p[i] = counter[i] / frequency;
		}
		for (int i = 0; i < p.length; i++) {
			if (p[i] == 0.0F)
				continue;
			entropy -= p[i] * (Math.log(p[i]) / Math.log(p.length));
		}
		return entropy;
	}

	public double thresholdSelection(double threshold, Double[] values, ThresholdType thr_alg) {
		Double mean = MathHelper.getInstance().calculateAverage(values);
		Double stdev = MathHelper.getInstance().calculateStdev(mean, values);
		Double[] q = MathHelper.getInstance().calculateQuartile(values);
		// threshold = mean + stdev; // Force using specified threshold
		switch (thr_alg) {
		case Mean:
			threshold = mean;
			logFile("Mean: " + threshold);
			break;
		case MeanMin:
			threshold = mean - stdev;
			logFile("Mean-: " + threshold);
			break;
		case MeanPlus:
			threshold = mean + stdev;
			logFile("Mean+: " + threshold);
			break;
		case Q1:
			threshold = q[0];
			logFile("Q1: " + q[0]);
			break;
		case Q2:
			threshold = q[1];
			logFile("Q2: " + q[1]);
			break;
		case Q3:
			threshold = q[2];
			logFile("Q3: " + q[2]);
			break;
		default:
			break;
		}
		return threshold;
	}

	public void logFile(String text) {
		String filename = LOG_FOLDER + timestamp + ".txt";
		try (BufferedWriter out = Files.newBufferedWriter(Paths.get(filename), Charset.forName("UTF-8"),
				StandardOpenOption.APPEND, StandardOpenOption.CREATE, StandardOpenOption.DSYNC)) {
			out.append(String.format("[%s] %s", new Date().toString(), text));
			out.newLine();

		} catch (IOException e) {
			if (Main.IS_DEBUG)
				System.err.println("FVSHelper.logFile exception");
			// e.printStackTrace();
		}
		System.err.println(text);
	}

	public void saveIntermediateInstances(Instances dataSet, String remark) {
		String filename = INTERMEDIATE_FOLDER + remark + ".arff";
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);
		try {
			saver.setFile(new File(filename));
			saver.writeBatch();
		} catch (IOException e) {
			if (Main.IS_DEBUG)
				System.err.println("FVSHelper.saveIntermediateInstances exception: " + e.getMessage());
			// e.printStackTrace();
		}
		System.out.println("Saved intermediate instances : " + filename);
	}

	public Instances loadIntermediateInstances(String remark) {
		Instances data = null;
		try {
			String filename = INTERMEDIATE_FOLDER + remark + ".arff";
			DataSource source = new DataSource(filename);
			data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			System.out.println("Loaded intermediate instances : " + filename);
		} catch (Exception e) {
			if (Main.IS_DEBUG)
				System.err.println("FVSHelper.loadIntermediateInstances exception: " + e.getMessage());
			// e.printStackTrace();
		}
		return data;
	}

	public boolean isIntermediateExist(String remark) {
		String filename = INTERMEDIATE_FOLDER + remark + ".arff";
		File f = new File(filename);
		return f.exists();
	}

	@SuppressWarnings({ "rawtypes" })
	public Map<String, List> initConfig() {
		Map<String, List> dict = new HashMap<>();
		String filename = CONFIG_FOLDER + "config.json";
		StringBuilder sb = new StringBuilder();
		try (BufferedReader in = Files.newBufferedReader(Paths.get(filename), Charset.forName("UTF-8"))) {
			String line = null;
			while ((line = in.readLine()) != null) {
				sb.append(line);
			}
		} catch (IOException e) {
			if (Main.IS_DEBUG)
				System.err.println("FVSHelper.initConfig exception");
		}
		// System.out.println(sb.toString());
		String[] configs = new String[] { "classifier", "discretization", "threshold", "preprocessing" };
		try {
			JSONObject rootObject = new JSONObject(sb.toString());
			for (String s : configs) {
				List<String> list = new ArrayList<>();
				JSONArray arr = rootObject.getJSONArray(s);
				for (int i = 0; i < arr.length(); i++) {
					String data = arr.getString(i);
					list.add(data);
				}
				dict.put(s, list);
			}
			/* Initialize folder(s) */
			JSONObject obj = null;
			obj = rootObject.getJSONObject("folder_setup");
			INTERMEDIATE_FOLDER = obj.get("intermediate").toString();
			if (!INTERMEDIATE_FOLDER.endsWith("\\") && !INTERMEDIATE_FOLDER.endsWith("/"))
				INTERMEDIATE_FOLDER = INTERMEDIATE_FOLDER + "\\";
			// System.out.println(INTERMEDIATE_FOLDER);
			// System.out.println(Arrays.asList(new
			// File(INTERMEDIATE_FOLDER).list()));
		} catch (JSONException e) {
			// JSON Parsing error
			e.printStackTrace();
		}
		return dict;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public List<ClassifierType> getClassifierType(Map<String, List> config) {
		List<ClassifierType> result = new ArrayList();
		List<String> list = config.get("classifier");
		for (String s : list) {
			if (s.equalsIgnoreCase("j48"))
				result.add(ClassifierType.J48);
			else if (s.equalsIgnoreCase("j48_pruned"))
				result.add(ClassifierType.J48_Pruned);
			else if (s.equalsIgnoreCase("jrip"))
				result.add(ClassifierType.JRip);
			else if (s.equalsIgnoreCase("jrip_pruned"))
				result.add(ClassifierType.JRip_Pruned);
			else if (s.equalsIgnoreCase("decision_stump"))
				result.add(ClassifierType.DecisionStump);
		}
		return result;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public List<DiscretizationType> getDiscretizationType(Map<String, List> config) {
		List<DiscretizationType> result = new ArrayList();
		List<String> list = config.get("discretization");
		for (String s : list) {
			if (s.equalsIgnoreCase("binning"))
				result.add(DiscretizationType.Binning);
			else if (s.equalsIgnoreCase("mdl"))
				result.add(DiscretizationType.MDL);
			else if (s.equalsIgnoreCase("frequency"))
				result.add(DiscretizationType.Frequency);
			else if (s.equalsIgnoreCase("none"))
				result.add(DiscretizationType.None);
		}
		return result;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public List<ThresholdType> getThresholdType(Map<String, List> config) {
		List<ThresholdType> result = new ArrayList();
		List<String> list = config.get("threshold");
		for (String s : list) {
			if (s.equalsIgnoreCase("na"))
				result.add(ThresholdType.NA);
			else if (s.equalsIgnoreCase("iteration"))
				result.add(ThresholdType.Iteration);
			else if (s.equalsIgnoreCase("q1"))
				result.add(ThresholdType.Q1);
			else if (s.equalsIgnoreCase("q2"))
				result.add(ThresholdType.Q2);
			else if (s.equalsIgnoreCase("q3"))
				result.add(ThresholdType.Q3);
			else if (s.equalsIgnoreCase("mean"))
				result.add(ThresholdType.Mean);
			else if (s.equalsIgnoreCase("meanmin"))
				result.add(ThresholdType.MeanMin);
			else if (s.equalsIgnoreCase("minplus"))
				result.add(ThresholdType.MeanPlus);
		}
		return result;
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public List<Preprocessing_Algorithm> getPreprocessing_Algorithm(Map<String, List> config) {
		List<Preprocessing_Algorithm> result = new ArrayList();
		List<String> list = config.get("preprocessing");
		for (String s : list) {
			if (s.equalsIgnoreCase("original"))
				result.add(Preprocessing_Algorithm.Original);
			else if (s.equalsIgnoreCase("random"))
				result.add(Preprocessing_Algorithm.Random);
			else if (s.equalsIgnoreCase("threshold"))
				result.add(Preprocessing_Algorithm.Threshold);
			else if (s.equalsIgnoreCase("correlation"))
				result.add(Preprocessing_Algorithm.Correlation);
			else if (s.equalsIgnoreCase("cfs"))
				result.add(Preprocessing_Algorithm.CFS);
			else if (s.equalsIgnoreCase("consistency"))
				result.add(Preprocessing_Algorithm.Consistency);
			else if (s.equalsIgnoreCase("projection"))
				result.add(Preprocessing_Algorithm.RandomProjection);
		}
		return result;
	}
}
