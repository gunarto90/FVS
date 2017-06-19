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
import java.util.Collection;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;

import edu.nctu.lalala.enums.ClassifierType;
import edu.nctu.lalala.enums.DiscretizationType;
import edu.nctu.lalala.enums.PreprocessingType;
import edu.nctu.lalala.enums.Preprocessing_Algorithm;
import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.CorrelationMatrix;
import edu.nctu.lalala.fvs.FV;
import edu.nctu.lalala.fvs.Value;
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
	private boolean ADD_NOISE = false;
	private int NOISE_LEVEL = 10; // 10 Percents
	private boolean IS_DEBUG = true;

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
				/* Normalize the value if correlation is a negative value */
				CM[i][j] = Math.abs(CM[i][j]);
				/* Symmetric property */
				/* Correlation between x and y is the same with y and x */
				CM[j][i] = CM[i][j];
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
	 * Calculate linear correlation between 2 columns <br/>
	 * Reference: https://en.wikipedia.org/wiki/Pearson_product-
	 * moment_correlation_coefficient
	 * 
	 * @param inst
	 * @param average
	 * @param x
	 * @param y
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

	public Instances transformInstances(Instances inst, Instances output, Map<FV, Collection<FV>> map,
			boolean removeInstance) {
		Set<FV> set = map.keySet();
		Double[] substitution = calculateAverage(inst);
		// Prepare the list
		// First level indicate which attribute the FVS resides in
		List<Map<Value, FV>> list = new ArrayList<>();
		for (int i = 0; i < inst.numAttributes(); i++) {
			list.add(new HashMap<Value, FV>());
		}
		// Build the data structure
		for (FV fv : set) {
			Value v = new Value(fv.getValue().toString());
			list.get(fv.getFeature()).put(v, fv);
		}
		for (int i = 0; i < inst.numInstances(); i++) {
			Instance instance = getFVSFilteredInstance(output, inst.instance(i), list, substitution, removeInstance, false);
			if (removeInstance && instance == null)
				continue;
			output.add(instance);
		}
		if (FVSHelper.getInstance().getDebugStatus()) {
			System.out.println("Input: " + inst.numInstances());
			System.out.println("Output: " + output.numInstances());
		}
		return output;
	}

	public Instances transformInstances(Instances inst, Instances output, Map<FV, Collection<FV>> map) {
		return transformInstances(inst, output, map, false);
	}
	
	public Instance getFVSFilteredInstance(Instances output, Instance old_inst, List<Map<Value, FV>> map,
			Double[] substitution, boolean removeInstance, boolean average) {
		double[] oldValues = old_inst.toDoubleArray();
		Instance instance = new DenseInstance(old_inst);
		int count_miss = 0;
		Random random = new Random();
		for (int i = 0; i < oldValues.length - 1; i++) {
			Value v = new Value(oldValues[i]);
			FV fv = map.get(i).getOrDefault(v, null);
			if (fv == null) {
				replaceValue(substitution, average, instance, i);
				count_miss++;
			} else if (old_inst.isMissing(i)) {
				count_miss++;
			} 
		}
		if (removeInstance) {
			/* Remove the instance using miss rate probability */
			double miss_rate = (double) count_miss / oldValues.length;
			if (miss_rate > random.nextFloat()) {
				instance = null;
			}
		}
		return instance;
	}

	public Instance getFVSFilteredInstance(Instances output, Instance old_inst, List<Map<Value, FV>> map,
			Double[] substitution) {
		return getFVSFilteredInstance(output, old_inst, map, substitution, false, false);
	}
	
	public void replaceValue(Double[] substitution, boolean average, Instance instance, int i) {
		/* Change with substitution */
		if (average)
			instance.setValue(i, substitution[i]);
		/* Change into missing */
		else
			instance.setMissing(i);
	}

	@SuppressWarnings({ "unchecked", "rawtypes" })
	public List<Double> generateEntropy(Map<FV, Collection<FV>> fv_list, int numInstances, int numOfClass) {
		List<Double> entropies = new ArrayList();
		Iterator<Entry<FV, Collection<FV>>> iterator = fv_list.entrySet().iterator();
		while (iterator.hasNext()) {
			Entry<FV, Collection<FV>> next = iterator.next();
			FV key = next.getKey();
			key.setFrequency((double) next.getValue().size() / numInstances);
			int[] counter = new int[key.getNumLabels()];
			for (FV fv : next.getValue()) {
				int idx = (int) fv.getLabel();
				counter[idx]++;
			}
			key.setEntropy(MathHelper.getInstance().calculateEntropy(counter, next.getValue().size(), numOfClass));
			entropies.add(key.getEntropy());
		}
		return entropies;
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
			if (IS_DEBUG)
				System.err.println("FVSHelper.logFile exception");
			// e.printStackTrace();
		}
		if (IS_DEBUG)
			System.out.println(text);
	}

	public void saveIntermediateInstances(Instances dataSet, String remark) {
		remark = remark.replace(".arff", "");
		String filename = INTERMEDIATE_FOLDER + remark + ".arff";
		File f = new File(filename);
		if (f.exists())
			return;
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);
		try {
			saver.setFile(f);
			saver.writeBatch();
		} catch (IOException e) {
			if (IS_DEBUG)
				System.err.println("FVSHelper.saveIntermediateInstances exception: " + e.getMessage());
			// e.printStackTrace();
		}

		if (IS_DEBUG)
			System.out.println("Saved intermediate instances : " + filename);
	}

	public Instances loadIntermediateInstances(String remark) {
		remark = remark.replace(".arff", "");
		Instances data = null;
		try {
			String filename = INTERMEDIATE_FOLDER + remark + ".arff";
			DataSource source = new DataSource(filename);
			data = source.getDataSet();
			if (data.classIndex() == -1)
				data.setClassIndex(data.numAttributes() - 1);
			if (IS_DEBUG)
				System.out.println("Loaded intermediate instances : " + filename);
		} catch (Exception e) {
			if (IS_DEBUG)
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

	@SuppressWarnings("rawtypes")
	public Map<String, List> initConfig() {
		return initConfig("config.json");
	}

	@SuppressWarnings({ "rawtypes" })
	public Map<String, List> initConfig(String configFilename) {
		System.err.println("Read config from: " + configFilename);
		Map<String, List> dict = new HashMap<>();
		String filename = CONFIG_FOLDER + configFilename;
		StringBuilder sb = new StringBuilder();
		try (BufferedReader in = Files.newBufferedReader(Paths.get(filename), Charset.forName("UTF-8"))) {
			String line = null;
			while ((line = in.readLine()) != null) {
				sb.append(line);
			}
		} catch (IOException e) {
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
			if (obj != null) {
				INTERMEDIATE_FOLDER = obj.get("intermediate").toString();
				if (!INTERMEDIATE_FOLDER.endsWith("\\") && !INTERMEDIATE_FOLDER.endsWith("/"))
					INTERMEDIATE_FOLDER = INTERMEDIATE_FOLDER + "\\";
			}
			/* Initialize add noise */
			obj = rootObject.getJSONObject("noise");
			if (obj != null) {
				ADD_NOISE = obj.getBoolean("enable_noise");
				NOISE_LEVEL = obj.getInt("noise_level");
			}
			/* Initialize debug status */
			try {
				IS_DEBUG = rootObject.getBoolean("debug");
			} catch (Exception ex) {
			}
			if (IS_DEBUG) {
				System.out.println("Debug mode is ON");
				System.out.println(String.format("Add noise: %s (%d percents)", ADD_NOISE, NOISE_LEVEL));
			}
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
			else if (s.equalsIgnoreCase("bayes"))
				result.add(ClassifierType.Bayes);
			else if (s.equalsIgnoreCase("logistic"))
				result.add(ClassifierType.Logistic);
			else if (s.equalsIgnoreCase("svm"))
				result.add(ClassifierType.SMO);
			else if (s.equalsIgnoreCase("instance"))
				result.add(ClassifierType.Instance);
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
			else if (s.equalsIgnoreCase("meanplus"))
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
			else if (s.equalsIgnoreCase("RandomFVS"))
				result.add(Preprocessing_Algorithm.FVS_Random);
			else if (s.equalsIgnoreCase("EntropyFVS"))
				result.add(Preprocessing_Algorithm.FVS_Entropy);
			else if (s.equalsIgnoreCase("CorrelationFVS"))
				result.add(Preprocessing_Algorithm.FVS_Correlation);
			else if (s.equalsIgnoreCase("RandomEntropyFVS"))
				result.add(Preprocessing_Algorithm.FVS_Random_Entropy);
			else if (s.equalsIgnoreCase("ProbabilisticFVS"))
				result.add(Preprocessing_Algorithm.FVS_Probabilistic);
			else if (s.equalsIgnoreCase("cfs"))
				result.add(Preprocessing_Algorithm.FS_CFS);
			else if (s.equalsIgnoreCase("consistency"))
				result.add(Preprocessing_Algorithm.FS_Consistency);
			else if (s.equalsIgnoreCase("projection"))
				result.add(Preprocessing_Algorithm.FT_RandomProjection);
			else if (s.equalsIgnoreCase("pca"))
				result.add(Preprocessing_Algorithm.FT_PCA);
			else if (s.equalsIgnoreCase("reservoir"))
				result.add(Preprocessing_Algorithm.IS_Reservoir);
			else if (s.equalsIgnoreCase("misclassified"))
				result.add(Preprocessing_Algorithm.IS_Misclassified);
		}
		return result;
	}

	public PreprocessingType getPreprocessType(Preprocessing_Algorithm p_alg) {
		PreprocessingType pt;
		switch (p_alg) {
		case Original:
			pt = PreprocessingType.None;
			break;
		case FVS_Correlation:
		case FVS_Random:
		case FVS_Entropy:
		case FVS_Random_Entropy:
		case FVS_Probabilistic:
			pt = PreprocessingType.FVS;
			break;
		case IS_Reservoir:
		case IS_Misclassified:
			pt = PreprocessingType.IS;
			break;
		case FT_RandomProjection:
		case FT_PCA:
		case FS_Consistency:
		case FS_CFS:
		case FS_CorrAttr:
		case FS_GainRatio:
		case FS_Kernel:
		case FS_GreedyStepwise:
		case FS_Relief:
		case FS_SymmetricUncertainty:
		case FS_Wrapper:
			pt = PreprocessingType.FS;
			break;
		default:
			pt = PreprocessingType.None;
		}
		return pt;
	}

	public boolean getAddNoise() {
		return ADD_NOISE;
	}

	public int getNoiseLevel() {
		return NOISE_LEVEL;
	}

	public boolean getDebugStatus() {
		return IS_DEBUG;
	}
}
