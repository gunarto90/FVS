package edu.nctu.lalala.main;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.nctu.lalala.enums.ClassifierType;
import edu.nctu.lalala.enums.DiscretizationType;
import edu.nctu.lalala.enums.PreprocessingType;
import edu.nctu.lalala.enums.Preprocessing_Algorithm;
import edu.nctu.lalala.enums.ThresholdType;
import edu.nctu.lalala.fvs.FVS_Filter;
import edu.nctu.lalala.util.FVSHelper;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ConsistencySubsetEval;
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
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.RandomProjection;

@SuppressWarnings("unused")
// Updated March 3rd, 2016
public class Main {

	public static final boolean IS_DEBUG = true;
	private static final boolean IS_LOG_INTERMEDIATE = true;
	private static final int NUMBER_OF_BINS = 10;
	private static final double[] DOUBLE_PARAMS = { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 };

	private static final String DEFAULT_DATASET_FOLDER = "dataset";
	private static final String NOMINAL_FOLDER = DEFAULT_DATASET_FOLDER + "/nominal/";
	private static final String NUMERIC_FOLDER = DEFAULT_DATASET_FOLDER + "/numeric/";
	private static final String TEST_FOLDER = DEFAULT_DATASET_FOLDER + "/run/";
	private static final String REPORT_FOLDER = "report" + "/";
	private static final String REPORT_HEADER = "User\tMethod\tClassifier\tDiscretization\tThreshold\tAccuracy\tModel ratio\tModel Size\tDouble param\n";
	/**
	 * Method - String<br/>
	 * Classification Algorithm - String<br/>
	 * Discretization - String<br/>
	 * Accuracy -Float<br/>
	 * Precision -Float<br/>
	 * Recall - Float<br/>
	 * Model size ratio - Float <br/>
	 */
	private static final String REPORT_FORMAT = "%s\t%s\t%s\t%s\t%s\t%.3f\t%.3f\t%d\t%.3f\n";

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
		PreprocessingType pt;
		System.out.println("Program Started");
		System.out.println(new Date());
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

		@SuppressWarnings("rawtypes")
		Map<String, List> config = FVSHelper.getInstance().initConfig();
		FVSHelper.getInstance().logFile(config.toString());
		FVSHelper.getInstance().logFile(Arrays.asList(folder.list()).toString());
		List<ClassifierType> cts = FVSHelper.getInstance().getClassifierType(config);
		List<DiscretizationType> dis = FVSHelper.getInstance().getDiscretizationType(config);
		List<ThresholdType> tts = FVSHelper.getInstance().getThresholdType(config);
		List<Preprocessing_Algorithm> fas = FVSHelper.getInstance().getPreprocessing_Algorithm(config);

		MathExpression mathexpr = new MathExpression();
		mathexpr.setIgnoreRange("13-14");
		mathexpr.setInvertSelection(true);
		try {
			mathexpr.setExpression("A*100000");
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		// For each file
		for (String datasetName : folder.list()) {
			try {
				// Load original data
				Instances data = null;
				// System.out.println("Load data finished");
				// For each discretization
				for (DiscretizationType dis_alg : dis) {
					// Create discretized set
					Instances discretized = null;
					boolean load_discretized = false;
					String discretized_cachename = String.format("%s_%s", datasetName, dis_alg.toString());
					if (FVSHelper.getInstance().isIntermediateExist(discretized_cachename))
						discretized = FVSHelper.getInstance().loadIntermediateInstances(discretized_cachename);
					if (discretized == null) {
						if (data == null) {
							// Only load data if necessary (no cache)
							data = loadData(lookupFolder + datasetName);
							data.deleteAttributeAt(0); // delete timestamp (NCTU, OPP) or user id (HAR)
							mathexpr.setInputFormat(data);
							data = Filter.useFilter(data, mathexpr);
						}
						discretized = discretize(data, dis_alg);
					} else
						load_discretized = true;
					FVSHelper.getInstance().logFile("Discertization: " + dis_alg);
					if (IS_LOG_INTERMEDIATE && !load_discretized)
						FVSHelper.getInstance().saveIntermediateInstances(discretized, discretized_cachename);
					// System.out.println("Discretization finished");
					boolean fs_cfs = false;
					boolean fs_consistency = false;
					boolean fs_projection = false;
					Map<ClassifierType, Double> originalModelSize = new HashMap<>();
					Map<ClassifierType, Integer> originalRule = new HashMap<>();
					/* Initialize the original model size and rule */
					for (ClassifierType type : cts) {
						FVSHelper.getInstance().logFile("Classifier: " + type);
						double modelSize = Double.NEGATIVE_INFINITY;
						modelSize = originalModelSize.getOrDefault(type, Double.NEGATIVE_INFINITY);
						if (modelSize == Double.NEGATIVE_INFINITY) {
							Classifier o_cl = null;
							Evaluation o_eval = null;
							// Build classifier based on original data
							o_cl = buildClassifier(discretized, type);
							// Evaluate the dataset
							o_eval = new Evaluation(discretized);
							// Cross validate dataset
							o_eval.crossValidateModel(o_cl, discretized, CROSS_VALIDATION, new Random(1));
							// Variables for evaluation
							double[] result = getModelSize(o_cl);
							modelSize = result[0];
							int rule = (int) result[1];
							if(type == ClassifierType.DecisionStump)
								rule = 1;
							originalModelSize.put(type, modelSize);
							originalRule.put(type, rule);
							writeReport(REPORT_FOLDER, datasetName, discretized.classIndex(), o_eval, o_cl, 1.0,
									1.0, "Original", type, dis_alg, "Original", rule);
						}
					}
					/* For each pre-processing */
					for (Preprocessing_Algorithm p_alg : fas) {
						FVSHelper.getInstance().logFile("Preprocessing: " + p_alg);
						pt = getPreprocessType(p_alg);
						// For each threshold type
						for (ThresholdType thr_alg : tts) {
							int run_preprocessing = 1;
							double modelSize = Double.NEGATIVE_INFINITY;
							int rule;
							// No random without iteration
							if (p_alg == Preprocessing_Algorithm.Random && thr_alg != ThresholdType.Iteration)
								continue;
							// Correlation and iteration cannot work together
							if (p_alg == Preprocessing_Algorithm.Correlation && thr_alg == ThresholdType.Iteration)
								continue;
							if (p_alg == Preprocessing_Algorithm.Threshold && thr_alg != ThresholdType.Iteration)
								run_preprocessing = 0;
							if (thr_alg == ThresholdType.NA)
								run_preprocessing = 0;
							/* Run FVS */
							if (pt == PreprocessingType.FVS) {
								double[] temp = null;
								if (run_preprocessing > 0) {
									temp = DOUBLE_PARAMS;
								} else {
									// Run once with the selected parameter
									temp = new double[] { 0.0 };
								}
								// Execution
								for (int i = 0; i < temp.length; i++) {
									Double double_param = temp[i];
									if (run_preprocessing <= 0) {
										FVSHelper.getInstance().logFile(datasetName + " : " + thr_alg);
									} else {
										FVSHelper.getInstance().logFile(datasetName + " : " + double_param);
									}
									if (p_alg == Preprocessing_Algorithm.Correlation
											&& (double_param == 1 || double_param == 0))
										continue;
									/*
									 * Filter the dataset using FVS
									 * algorithms
									 */
									Instances filtered = null;
									filtered = featureValueSelection(discretized, p_alg, thr_alg,
											discretized.numInstances(), double_param);
									/*
									 * Build classifier using filtered data
									 */
									for (ClassifierType type : cts) {
										if(type == ClassifierType.DecisionStump)
											continue;
										FVSHelper.getInstance().logFile("Classifier: " + type);
										Classifier f_cl = buildClassifier(filtered, type);
										// Evaluate the dataset
										Evaluation f_eval = new Evaluation(filtered);
										// Cross validate dataset
										f_eval.crossValidateModel(f_cl, filtered, CROSS_VALIDATION, new Random(1));
										// Compare model size
										double[] result = getModelSize(f_cl);
										modelSize = result[0] / originalModelSize.get(type);
										rule = (int) result[1];
										writeReport(REPORT_FOLDER, datasetName, filtered.classIndex(), f_eval, f_cl,
												modelSize, double_param, p_alg, type, dis_alg, thr_alg, rule);
									}
									filtered.delete();
								}
								System.gc();
							}
							/* Run Other Preprocessings */
							else {
								if ((fs_cfs && p_alg == Preprocessing_Algorithm.CFS)
										|| (fs_consistency && p_alg == Preprocessing_Algorithm.Consistency)
										|| (fs_projection && p_alg == Preprocessing_Algorithm.RandomProjection))
									continue;
								Instances filtered = null;
								filtered = applySelection(discretized, p_alg);
								/*
								 * Build classifier based on filtered data
								 */
								for (ClassifierType type : cts) {
									if(type == ClassifierType.DecisionStump)
										continue;
									FVSHelper.getInstance().logFile("Classifier: " + type);
									Classifier f_cl = buildClassifier(filtered, type);
									// Evaluate the dataset
									Evaluation f_eval = new Evaluation(filtered);
									// Cross validate dataset
									f_eval.crossValidateModel(f_cl, filtered, CROSS_VALIDATION, new Random(1));
									double[] result = getModelSize(f_cl);
									modelSize = result[0] / originalModelSize.get(type);
									rule = (int) result[1];
									writeReport(REPORT_FOLDER, datasetName, filtered.classIndex(), f_eval, f_cl,
											modelSize, 0.0, p_alg, type, dis_alg, thr_alg, rule);
								}
								filtered.delete();
								System.gc();
								// No need to loop over various "threshold"
								if (p_alg == Preprocessing_Algorithm.CFS)
									fs_cfs = true;
								else if (p_alg == Preprocessing_Algorithm.Consistency)
									fs_consistency = true;
								else if (p_alg == Preprocessing_Algorithm.RandomProjection)
									fs_projection = true;
							}
						} // For each threshold type
					} // For each preprocessing algorithm
					System.out.println();
					if (data != null)
						data.delete();
					System.gc();
				} // For each discretization
			} // Try for each file
			catch (Exception e) {
				if (IS_DEBUG)
					e.printStackTrace();
				FVSHelper.getInstance().logFile(e.getMessage());
			}
		} // For each file

		System.out.println("Program finished");
		System.out.println(new Date());
	}

	private double[] getModelSize(Classifier o_cl) {
		double[] result = new double[2];
		double modelSize = 0.0;
		double rule = 0.0;
		try {
			J48 a = (J48) o_cl;
			modelSize = a.measureNumLeaves();
			rule = a.measureNumRules();
		} catch (Exception e) {

		}
		if (modelSize == 0.0) {
			try {
				JRip a = (JRip) o_cl;
				modelSize = a.getRuleset().size();
				rule = modelSize;
			} catch (Exception e) {

			}
		}
		result[0] = modelSize;
		result[1] = rule;
		return result;
	}

	private PreprocessingType getPreprocessType(Preprocessing_Algorithm p_alg) {
		PreprocessingType pt;
		switch (p_alg) {
		case Correlation:
		case Random:
		case Threshold:
		case Original:
			pt = PreprocessingType.FVS;
			break;
		case IS:
			pt = PreprocessingType.IS;
			break;
		case CFS:
		case RandomProjection:
		case Consistency:
		case RELLAGS:
			pt = PreprocessingType.FS;
			break;
		default:
			pt = PreprocessingType.FVS;
		}
		return pt;
	}

	private Instances loadData(String file) throws Exception {
		DataSource source;

		source = new DataSource(file);

		if (IS_DEBUG) {
			FVSHelper.getInstance().logFile(file);
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
			((JRip) c).setUsePruning(false);
			break;
		case JRip_Pruned:
			c = new JRip();
			((JRip) c).setUsePruning(true);
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
		if (type == DiscretizationType.None) {
			return data;
		} else if (type == DiscretizationType.Binning) {
			filter = new weka.filters.unsupervised.attribute.Discretize();
			((weka.filters.unsupervised.attribute.Discretize) filter).setBins(NUMBER_OF_BINS);
		} else if (type == DiscretizationType.Frequency) {
			filter = new weka.filters.unsupervised.attribute.Discretize();
			((weka.filters.unsupervised.attribute.Discretize) filter).setUseEqualFrequency(true);
		} else if (type == DiscretizationType.MDL)
			filter = new weka.filters.supervised.attribute.Discretize();
		try {
			if (filter == null)
				return null;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
			FVSHelper.getInstance().logFile(e.getMessage());
		}
		return result;
	}

	private Instances featureValueSelection(Instances data, Preprocessing_Algorithm algo, ThresholdType thr_alg,
			int numInstances, Double... params) {
		Instances result = null;
		Filter filter = new FVS_Filter(algo, thr_alg, numInstances, params);
		try {
			if (filter == null)
				return data;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
			FVSHelper.getInstance().logFile(e.getMessage());
		}
		return result;
	}

	private Instances applySelection(Instances data, Preprocessing_Algorithm algo) {
		Instances result = null;
		Filter filter = null;
		switch (algo) {
		case CFS:
			filter = new AttributeSelection();
			AttributeSelection temp = (AttributeSelection) filter;
			CfsSubsetEval cfs = new CfsSubsetEval();
			temp.setEvaluator(cfs);
			break;
		case Consistency:
			filter = new AttributeSelection();
			temp = (AttributeSelection) filter;
			ConsistencySubsetEval cs = new ConsistencySubsetEval();
			temp.setEvaluator(cs);
			break;
		case RandomProjection:
			filter = new RandomProjection();
			break;
		default:
			break;
		}
		try {
			if (filter == null)
				return data;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
		} catch (Exception e) {
			e.printStackTrace();
			FVSHelper.getInstance().logFile(e.getMessage());
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
		String uid = datasetName;
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
		String threshold = double_param + "";
		if (threshold.trim().equalsIgnoreCase("nan"))
			threshold = params[3].toString();
		fileWriter.append(
				String.format(REPORT_FORMAT, uid, params[0].toString(), params[1].toString(), params[2].toString(),
						params[3].toString(), calculateAccuracy(TP, TN, FP, FN), model_ratio, params[4], double_param));
		fileWriter.close();
		// fileWriterCopy.append(String.format(REPORT_FORMAT,
		// params[0].toString(), params[1].toString(), params[2].toString(),
		// params[3].toString(), calculateAccuracy(TP, TN, FP, FN), model_ratio,
		// params[4], double_param));
		// fileWriterCopy.close();
	}

}
