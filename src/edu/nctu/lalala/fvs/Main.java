package edu.nctu.lalala.fvs;

import java.io.File;
import java.io.FileWriter;
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
	private static final String TEST_FOLDER = DEFAULT_DATASET_FOLDER + "/htc/";
	private static final String REPORT_FOLDER = "report" + "/";
	private static final String REPORT_HEADER = "Method\tClassifier\tDiscretization\tAccuracy\tPrecision\tRecall\tModel ratio\tDouble param\n";
	/**
	 * Method - String<br/>
	 * Classification Algorithm - String<br/>
	 * Discretization - String<br/>
	 * Accuracy -Float<br/>
	 * Precision -Float<br/>
	 * Recall - Float<br/>
	 * Model size ratio - Float <br/>
	 */
	private static final String REPORT_FORMAT = "%s\t%s\t%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n";

	private static final int RUN_REPETITION = 10;

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
		DiscretizationType dis_alg = DiscretizationType.Binning;
		FVS_Algorithm fvs_alg = FVS_Algorithm.Threshold;

		for (String datasetName : folder.list()) {
			try {
				double double_param = 0.0;
				// Load original data
				Instances data = loadData(lookupFolder + datasetName);
				// Create discretized set
				Instances discretized = discretize(data, dis_alg);
				// Build classifier based on original data
				Classifier o_cl = buildClassifier(discretized, type);
				// Evaluate the dataset
				Evaluation o_eval = new Evaluation(discretized);
				// Cross validate dataset
				o_eval.crossValidateModel(o_cl, discretized, CROSS_VALIDATION, new Random(1));
//				printEvaluation(o_eval, null, discretized.classIndex(), "Original");
				writeReport(REPORT_FOLDER, datasetName, discretized.classIndex(), o_eval, o_cl,
						1, double_param, "Original", type, dis_alg);
				// System.out.println(o_cl.toString());
				for (int i = 0; i <= RUN_REPETITION; i++) {
					double_param = (double) i / RUN_REPETITION;
					System.out.println(datasetName + " : " + double_param);
					// Filter dataset using FVS algorithm
					Instances filtered = featureValueSelection(discretized, fvs_alg, discretized.numInstances(),
							double_param);
					// Build classifier based on filtered data
					Classifier f_cl = buildClassifier(filtered, type);
					// Evaluate the dataset
					Evaluation f_eval = new Evaluation(filtered);
					// Cross validate dataset
					f_eval.crossValidateModel(f_cl, filtered, CROSS_VALIDATION, new Random(1));
					// if (IS_DEBUG)
					// System.out.println(cl.toString());
//					printEvaluation(f_eval, null, filtered.classIndex(), "Filtered");
					// Compare model size
					J48 a = (J48) o_cl;
					J48 b = (J48) f_cl;
//					writeReport(REPORT_FOLDER, datasetName, filtered.classIndex(), f_eval, f_cl,
//							(double) f_cl.toString().length() / base_model_size, double_param, fvs_alg, type, dis_alg);
					writeReport(REPORT_FOLDER, datasetName, filtered.classIndex(), f_eval, f_cl,
							(double) b.measureNumLeaves() / a.measureNumLeaves(), double_param, fvs_alg, type, dis_alg);
				}
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
			((J48)c).setUnpruned(true);
			break;
		case JRip:
			c = new JRip();
			break;
		case SMO:
			c = new SMO();
			break;
		default:
			c = new J48();
			((J48)c).setUnpruned(true);
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

		// System.out.println(eval.toSummaryString());

		System.out.println("Total instances: " + total);
		System.out.println("Correct: " + TP);
		System.out.println("Accuracy: " + calculateAccuracy(TP, TN, FP, FN));
		System.out.println("Precision: " + calculatePrecision(TP, FP));
		System.out.println("Recall: " + calculateRecall(TP, FN));
		for (String s : params) {
			System.out.println(s);
		}
		System.out.println();

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

	private Instances featureValueSelection(Instances data, FVS_Algorithm algo, int numInstances, Double... params) {
		Instances result = null;
		Filter filter = new FVS(algo, numInstances, params);
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

	private void writeReport(String folder, String datasetName, int classIndex, Evaluation eval, Classifier cl,
			double model_ratio, double double_param, Object... params) throws IOException {
		if (!folder.endsWith("/"))
			folder = folder + "/";
		if (datasetName.contains(".")) {
			datasetName = datasetName.split("\\.")[0];
		}
		datasetName = datasetName + ".txt";
		File ffolder = new File(folder);
		ffolder.mkdirs();
		File f = new File(folder + datasetName);
		boolean new_file = f.createNewFile();
		FileWriter fileWriter = new FileWriter(f, true);
		if (new_file)
			fileWriter.write(REPORT_HEADER);
		double TP, TN, FP, FN;
		double total = eval.numInstances();
		TP = eval.correct();
		TN = eval.incorrect();
		FP = eval.numFalsePositives(classIndex);
		FN = eval.numFalsePositives(classIndex);
		fileWriter.append(String.format(REPORT_FORMAT, params[0].toString(), params[1].toString(), params[2].toString(),
				calculateAccuracy(TP, TN, FP, FN), calculatePrecision(TP, FP), calculateRecall(TP, FN), model_ratio,
				double_param));
		fileWriter.close();
	}

}
