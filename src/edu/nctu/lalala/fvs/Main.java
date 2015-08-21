package edu.nctu.lalala.fvs;

import java.io.File;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

enum ClassifierType {
	J48, JRip, SMO
}

enum DiscretizationType {
	Binning, MDL
}

@SuppressWarnings("unused")
public class Main {

	private static final boolean IS_DEBUG = true;

	private static final String DEFAULT_DATASET_FOLDER = "dataset";
	private static final String NOMINAL_FOLDER = DEFAULT_DATASET_FOLDER + "/nominal/";
	private static final String NUMERIC_FOLDER = DEFAULT_DATASET_FOLDER + "/numeric/";
	private static final String TEST_FOLDER = DEFAULT_DATASET_FOLDER + "/test/";

	private int CROSS_VALIDATION = 10;

	/**
	 * static Singleton instance
	 */
	private static Main instance;

	/**
	 * Private constructor for singleton
	 */
	private Main() {
	}

	/**
	 * Static getter method for retrieving the singleton instance
	 */
	public static Main getInstance() {
		if (instance == null) {
			instance = new Main();
		}
		return instance;
	}

	public static void main(String[] args) {
		getInstance().program(args);
	}

	/**
	 * Real main method (without static)
	 * 
	 * @param args
	 */
	private void program(String[] args) {
		/************************/
		/* MAIN_PROGRAM IS HERE */
		/************************/
		String lookupFolder = TEST_FOLDER;

		// Init using args if possible
		if (args.length == 1) {
			lookupFolder = args[0];
		} else if (args.length == 2) {
			lookupFolder = args[0];
			try {
				CROSS_VALIDATION = Integer.parseInt(args[1]);
			} catch (Exception e) {
			}
		}

		if (!lookupFolder.endsWith("/"))
			lookupFolder = lookupFolder + "/";

		File folder = new File(lookupFolder);
		if (IS_DEBUG)
			System.err.println(lookupFolder);

		ClassifierType type = ClassifierType.J48;

		for (String f : folder.list()) {
			try {
				// Load original data
				Instances data = loadData(lookupFolder + f);
				// Create discretized set
				Instances discretized = discretize(data, DiscretizationType.Binning);
				// Filter dataset using FVS algorithm
				Instances filtered = featureValueSelection(discretized, FVS_Algorithm.Threshold);
				// Build classifier based on original data
				Classifier o_cl = buildClassifier(discretized, type);
				// Build classifier based on filtered data
				Classifier f_cl = buildClassifier(filtered, type);
				// // Evaluate the dataset
				Evaluation o_eval = new Evaluation(discretized);
				Evaluation f_eval = new Evaluation(filtered);
				// // Cross validate dataset
				o_eval.crossValidateModel(o_cl, discretized, CROSS_VALIDATION, new Random(1));
				f_eval.crossValidateModel(f_cl, filtered, CROSS_VALIDATION, new Random(1));
				// if (IS_DEBUG)
				// System.out.println(cl.toString());
				printEvaluation(o_eval, null, discretized.classIndex());
				printEvaluation(f_eval, null, filtered.classIndex());
				// Compare model size
				compareModelSize(o_cl, f_cl);
				System.out.println(o_cl.toString());
				System.out.println(f_cl.toString());
			} catch (Exception e) {
				if (IS_DEBUG)
					e.printStackTrace();
			}
		}
		System.out.println("Program finished");
	}

	private Instances loadData(String file) throws Exception {
		DataSource source;

		source = new DataSource(file);

		if (IS_DEBUG)
			System.err.println(file);
		Instances data = source.getDataSet();
		// setting class attribute if the data format does not provide
		// this information. For example, the XRFF format saves the
		// class attribute information as well
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	private Classifier buildClassifier(Instances data, ClassifierType type) throws Exception {
		Classifier c = null;
		if (type == null)
			type = ClassifierType.J48;
		switch (type) {
		case J48:
			c = new J48();
			break;
		case JRip:
			c = new JRip();
			break;
		case SMO:
			c = new SMO();
			break;
		default:
			c = new J48();
			break;
		}
		c.buildClassifier(data);
		return c;
	}

	private double calculatePrecision(double TP, double FP) {
		return TP / (TP + FP);
	}

	private double calculateRecall(double TP, double FN) {
		return TP / (TP + FN);
	}

	private double calculateAccuracy(double TP, double TN, double FP, double FN) {
		return (TP) / (TP + TN + FP + FN);
	}

	private void printEvaluation(Evaluation eval, String outputFile, int classIndex, String... params) {
		double TP, TN, FP, FN;
		double total = eval.numInstances();
		TP = eval.correct();
		TN = eval.incorrect();
		FP = eval.numFalsePositives(classIndex);
		FN = eval.numFalsePositives(classIndex);

		System.out.println(eval.toSummaryString());

		System.out.println("Total instances: " + total);
		System.out.println("Correct: " + TP);
		System.out.println("Accuracy: " + calculateAccuracy(TP, TN, FP, FN));
		System.out.println("Precision: " + calculatePrecision(TP, FP));
		System.out.println("Recall: " + calculateRecall(TP, FN));

		if (outputFile != null) {
			// Write to file
		}
	}

	private Instances discretize(Instances data) {
		return discretize(data, DiscretizationType.Binning);
	}

	private Instances discretize(Instances data, DiscretizationType type) {
		Instances result = null;
		Filter filter = null;
		if (type == DiscretizationType.Binning)
			filter = new weka.filters.unsupervised.attribute.Discretize();
		else if (type == DiscretizationType.MDL)
			filter = new weka.filters.supervised.attribute.Discretize();
		try {
			if (filter == null)
				return null;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	private Instances featureValueSelection(Instances data, FVS_Algorithm algo) {
		Instances result = null;
		Filter filter = new FVS(algo);
		try {
			if (filter == null)
				return null;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return result;
	}

	public void compareModelSize(Classifier original, Classifier filtered) throws IOException {
		System.out.println("Ori_length: " + original.toString().length());
		System.out.println("Filtered_length: " + filtered.toString().length());
		// File f_ori = File.createTempFile("ccc_original", "weka_model");
		// File f_filtered = File.createTempFile("ccc_filtered", "weka_model");
		// try {
		// weka.core.SerializationHelper.write(f_ori.getAbsolutePath(),
		// original);
		// weka.core.SerializationHelper.write(f_filtered.getAbsolutePath(),
		// filtered);
		// System.out.println(f_ori.getAbsolutePath());
		// System.out.println(f_filtered.getAbsolutePath());
		// System.out.println("Ori_length: "+f_ori.length());
		// System.out.println("Filtered_length: "+f_filtered.length());
		// } catch (Exception e) {
		// e.printStackTrace();
		// }
	}

}
