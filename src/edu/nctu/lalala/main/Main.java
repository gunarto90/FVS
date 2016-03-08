package edu.nctu.lalala.main;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Date;

import edu.nctu.lalala.enums.ClassifierType;
import edu.nctu.lalala.enums.DiscretizationType;
import edu.nctu.lalala.enums.FVS_Algorithm;
import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.FVS_Filter;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

@SuppressWarnings("unused")
// Updated March 3rd, 2016
public class Main {

	private static final boolean IS_DEBUG = true;

	private static final String DEFAULT_DATASET_FOLDER = "dataset";
	private static final String NOMINAL_FOLDER = DEFAULT_DATASET_FOLDER + "/nominal/";
	private static final String NUMERIC_FOLDER = DEFAULT_DATASET_FOLDER + "/numeric/";
	private static final String TEST_FOLDER = DEFAULT_DATASET_FOLDER + "/run/";
	private static final String REPORT_FOLDER = "report" + "/";
	private static final String REPORT_HEADER = "Method\tClassifier\tDiscretization\tThreshold\tAccuracy\tModel ratio\tModel Size\tDouble param\n";
	/**
	 * Method - String<br/>
	 * Classification Algorithm - String<br/>
	 * Discretization - String<br/>
	 * Accuracy -Float<br/>
	 * Precision -Float<br/>
	 * Recall - Float<br/>
	 * Model size ratio - Float <br/>
	 */
	private static final String REPORT_FORMAT = "%s\t%s\t%s\t%s\t%.3f\t%.3f\t%d\t%.3f\n";

	private static int RUN_REPETITION = 10;

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

		// TODO Change Variables here
//		ClassifierType[] cts = { ClassifierType.J48, ClassifierType.J48_Pruned, ClassifierType.DecisionStump };
//		DiscretizationType[] dis = { DiscretizationType.Binning, DiscretizationType.MDL };
//		ThresholdType[] tts = { ThresholdType.Iteration, ThresholdType.Q1, ThresholdType.Q2, ThresholdType.Q3,
//				ThresholdType.Mean, ThresholdType.MeanPlus, ThresholdType.MeanMin };
//		FVS_Algorithm[] fas = { FVS_Algorithm.Threshold, FVS_Algorithm.Random, FVS_Algorithm.Correlation };

		// Custom
		ClassifierType[] cts = { ClassifierType.J48, ClassifierType.J48_Pruned};
		DiscretizationType[] dis = { DiscretizationType.Binning, DiscretizationType.MDL};
		ThresholdType[] tts = { ThresholdType.Q2 };
		FVS_Algorithm[] fas = { FVS_Algorithm.Correlation };

		// For each classifier
		for (ClassifierType type : cts) {
			// For each fvs
			for (FVS_Algorithm fvs_alg : fas) {
				// For each discretization
				for (DiscretizationType dis_alg : dis) {
					// For each threshold type
					for (ThresholdType thr_alg : tts) {
						// For each file
						for (String datasetName : folder.list()) {
							try {
								if (fvs_alg == FVS_Algorithm.Random && thr_alg != ThresholdType.Iteration)
									continue;
								if (fvs_alg == FVS_Algorithm.Correlation && thr_alg == ThresholdType.Iteration)
									continue;
								int run = RUN_REPETITION;
								double double_param = 0.0;
								if (fvs_alg == FVS_Algorithm.Correlation && (double_param == 1 || double_param == 0))
									continue;
								// Load original data
								Instances data = loadData(lookupFolder + datasetName);
								data.deleteAttributeAt(0); // delete timestamp
								// System.out.println("Load data finished");
								// Create discretized set
								Instances discretized = discretize(data, dis_alg);
								// System.out.println("Discretization
								// finished");
								// Build classifier based on original data
								Classifier o_cl = buildClassifier(discretized, type);
								// Evaluate the dataset
								Evaluation o_eval = new Evaluation(discretized);
								// Cross validate dataset
								o_eval.crossValidateModel(o_cl, discretized, CROSS_VALIDATION, new Random(1));
								double modelSize = 0;
								int rule1, rule2;
								rule1 = rule2 = 0;
								J48 a = null, b = null;
								try {
									a = (J48) o_cl;
									modelSize = a.measureNumLeaves();
									rule1 = (int) a.measureNumRules();
								} catch (Exception e) {

								}
								// printEvaluation(o_eval, null,
								// discretized.classIndex(),
								// "Original");
								writeReport(REPORT_FOLDER, datasetName, discretized.classIndex(), o_eval, o_cl,
										modelSize, double_param, FVS_Algorithm.Original, type, dis_alg, thr_alg, rule1);
								// System.out.println(o_cl.toString());
								if (fvs_alg == FVS_Algorithm.Threshold && thr_alg != ThresholdType.Iteration)
									run = 0; // No need to iterate
								if (type == ClassifierType.DecisionStump)
									run = -1;
								for (int i = run; i >= 0; i--) {
									double_param = (double) i / run;
									System.out.println(datasetName + " : " + double_param);
									// Filter dataset using FVS algorithm
									Instances filtered = featureValueSelection(discretized, fvs_alg, thr_alg,
											discretized.numInstances(), double_param);
									// Build classifier based on filtered data
									Classifier f_cl = buildClassifier(filtered, type);
									// Evaluate the dataset
									Evaluation f_eval = new Evaluation(filtered);
									// Cross validate dataset
									f_eval.crossValidateModel(f_cl, filtered, CROSS_VALIDATION, new Random(1));
									// if (IS_DEBUG)
									// System.out.println(cl.toString());
									// printEvaluation(f_eval, null,
									// filtered.classIndex(),
									// "Filtered");
									// Compare model size

									try {
										b = (J48) f_cl;
										modelSize = (double) b.measureNumLeaves() / a.measureNumLeaves();
										rule2 = (int) b.measureNumRules();
									} catch (Exception e) {

									}
									// writeReport(REPORT_FOLDER, datasetName,
									// filtered.classIndex(), f_eval, f_cl,
									// (double) f_cl.toString().length() /
									// base_model_size,
									// double_param, fvs_alg, type, dis_alg);
									writeReport(REPORT_FOLDER, datasetName, filtered.classIndex(), f_eval, f_cl,
											modelSize, double_param, fvs_alg, type, dis_alg, thr_alg, rule2);
									filtered.delete();
									System.gc();
								}
								System.out.println();
								data.delete();
								System.gc();
							} catch (Exception e) {
								if (IS_DEBUG)
									e.printStackTrace();
							}
						} // For each file
					} // For each threshold type
				} // For each discretization
			} // For each fvs
		} // For each classifier

		System.out.println("Program finished");
	}

	private Instances loadData(String file) throws Exception {
		DataSource source;

		source = new DataSource(file);

		if (IS_DEBUG)
		{
			System.err.println(file);
			System.err.println(new Date().toString());
		}
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
			((J48) c).setUnpruned(true);
			break;
		case J48_Pruned:
			c = new J48();
			((J48) c).setUnpruned(false);
			break;
		case JRip:
			c = new JRip();
			break;
		case SMO:
			c = new SMO();
			break;
		case DecisionStump:
			c = new DecisionStump();
			break;
		default:
			c = new J48();
			((J48) c).setUnpruned(true);
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

	private Instances featureValueSelection(Instances data, FVS_Algorithm algo, ThresholdType thr_alg, int numInstances,
			Double... params) {
		Instances result = null;
		Filter filter = new FVS_Filter(algo, thr_alg, numInstances, params);
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
		// String copyName = datasetName + params[0].toString() + "_" +
		// params[1].toString() + "_" + params[2].toString() + "_" +
		// params[3].toString() + ".txt";
		datasetName = datasetName + ".txt";
		File ffolder = new File(folder);
		ffolder.mkdirs();
		File f = new File(folder + datasetName);
		// File f_copy = new File(folder + copyName);
		boolean new_file = f.createNewFile();
		FileWriter fileWriter = new FileWriter(f, true);
		// FileWriter fileWriterCopy = new FileWriter(f_copy, true);
		if (new_file) {
			fileWriter.write(REPORT_HEADER);
		}
		double TP, TN, FP, FN;
		double total = eval.numInstances();
		TP = eval.correct();
		TN = eval.incorrect();
		FP = eval.numFalsePositives(classIndex);
		FN = eval.numFalseNegatives(classIndex);
		// fileWriter.append(String.format(REPORT_FORMAT, params[0].toString(),
		// params[1].toString(), params[2].toString(),
		// calculateAccuracy(TP, TN, FP, FN), calculatePrecision(TP, FP),
		// calculateRecall(TP, FN), model_ratio,
		// double_param));
		fileWriter.append(String.format(REPORT_FORMAT, params[0].toString(), params[1].toString(), params[2].toString(),
				params[3].toString(), calculateAccuracy(TP, TN, FP, FN), model_ratio, params[4], double_param));
		fileWriter.close();
		// fileWriterCopy.append(String.format(REPORT_FORMAT,
		// params[0].toString(), params[1].toString(), params[2].toString(),
		// params[3].toString(), calculateAccuracy(TP, TN, FP, FN), model_ratio,
		// params[4], double_param));
		// fileWriterCopy.close();
	}

}
