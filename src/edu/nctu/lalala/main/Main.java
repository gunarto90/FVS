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
import edu.nctu.lalala.fvs.algorithm.EntropyFVS;
import edu.nctu.lalala.fvs.evaluation.FVSEvaluation;
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
import weka.filters.unsupervised.attribute.AddNoise;
import weka.filters.unsupervised.attribute.MathExpression;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RandomProjection;
import weka.filters.unsupervised.instance.RemoveMisclassified;
import weka.filters.unsupervised.instance.ReservoirSample;

@SuppressWarnings("unused")
// Updated March 3rd, 2016
public class Main {
	private static final boolean IS_LOG_INTERMEDIATE = true;
	private static final double[] DOUBLE_PARAMS = { 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 };

	private static final String DEFAULT_DATASET_FOLDER = "dataset";
	private static final String NOMINAL_FOLDER = DEFAULT_DATASET_FOLDER + "/nominal/";
	private static final String NUMERIC_FOLDER = DEFAULT_DATASET_FOLDER + "/numeric/";
	private static final String TEST_FOLDER = DEFAULT_DATASET_FOLDER + "/run/";
	private static final String REPORT_FOLDER = "report" + "/";
	private static final String REPORT_HEADER = "User\tMethod\tClassifier\tDiscretization\tThreshold\tAccuracy\t#Rules\tDouble param\tFile Size\tRunning Time(ms)\tMemory Usage(KB)\tNoise\n";
	/**
	 * Method - String<br/>
	 * Classification Algorithm - String<br/>
	 * Discretization - String<br/>
	 * Threshold - String<br/>
	 * Accuracy -Float<br/>
	 * Model Size -int<br/>
	 * Double Param - Float<br/>
	 */
	private static final String REPORT_FORMAT = "%s\t%s\t%s\t%s\t%s\t%.3f\t%d\t%.3f\t%d\t%.3f\t%d\t%d\n";

	private int CROSS_VALIDATION = 3;

	private static int NUMBER_OF_BINS = 10;

	public static int NUMBER_OF_CLASS = 6;

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
		String customConfigFile = null;

		// Init using args if possible
		if (args.length == 0) {
			customConfigFile = "config_test.json";
		} else if (args.length == 1) {
			lookupFolder = args[0];
		} else if (args.length == 2) {
			lookupFolder = args[0];
			try {
				CROSS_VALIDATION = Integer.parseInt(args[1]);
			} catch (Exception e) {
			}
		} else if (args.length == 3) {
			lookupFolder = args[0];
			try {
				CROSS_VALIDATION = Integer.parseInt(args[1]);
			} catch (Exception e) {
			}
			customConfigFile = args[2];
		}

		if (!lookupFolder.endsWith("/"))
			lookupFolder = lookupFolder + "/";

		File folder = new File(lookupFolder);
		if (FVSHelper.getInstance().getDebugStatus())
			System.err.println(lookupFolder);

		@SuppressWarnings("rawtypes")
		Map<String, List> config;
		if (customConfigFile == null)
			config = FVSHelper.getInstance().initConfig();
		else
			config = FVSHelper.getInstance().initConfig(customConfigFile);
		FVSHelper.getInstance().logFile(config.toString());
		FVSHelper.getInstance().logFile(Arrays.asList(folder.list()).toString());
		FVSHelper.getInstance().logFile("Cross validation: " + CROSS_VALIDATION);
		List<ClassifierType> cts = FVSHelper.getInstance().getClassifierType(config);
		List<DiscretizationType> dis = FVSHelper.getInstance().getDiscretizationType(config);
		List<ThresholdType> tts = FVSHelper.getInstance().getThresholdType(config);
		List<Preprocessing_Algorithm> fas = FVSHelper.getInstance().getPreprocessing_Algorithm(config);

		MathExpression mathexpr = new MathExpression();
		mathexpr.setIgnoreRange("4-6,13-14");
		mathexpr.setInvertSelection(true);
		try {
			mathexpr.setExpression("A*100000");
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		// For each file
		for (String datasetName : folder.list()) {
			if (!datasetName.endsWith(".arff"))
				continue;
			try {
				// Load original data
				Instances data = null;
				// For each discretization
				for (DiscretizationType dis_alg : dis) {
					// Create discretized set
					Instances discretized = null;
					boolean load_discretized = false;
					String noise_name = "";
					String discretized_cachename = String.format("%s_%s", datasetName.replace(".arff", ""),
							dis_alg.toString());
					if (FVSHelper.getInstance().isIntermediateExist(discretized_cachename))
						discretized = FVSHelper.getInstance().loadIntermediateInstances(discretized_cachename);
					if (discretized == null) {
						if (data == null || data.numInstances() <= 0) {
							// Only load data if necessary (no cache)
							data = loadData(lookupFolder + datasetName);
							/*
							 * Delete timestamp (NCTU and OPP) or user id (HAR)
							 */
							data.deleteAttributeAt(0);
							/* Adjust GPS_X and GPS_Y in NCTU dataset */
							if (datasetName.contains("agg")) {
								mathexpr.setInputFormat(data);
								data = Filter.useFilter(data, mathexpr);
							}
						}
						discretized = discretize(data, dis_alg);
					} else
						load_discretized = true;
					FVSHelper.getInstance().logFile("Discertization: " + dis_alg);
					if (IS_LOG_INTERMEDIATE && !load_discretized)
						FVSHelper.getInstance().saveIntermediateInstances(discretized, discretized_cachename);
					FVSHelper.getInstance().logFile(String.format("File %s has been loaded completely", datasetName));
					if (data != null)
						data.delete();
					data = null;
					NUMBER_OF_CLASS = discretized.numClasses();

					/* For each pre-processing */
					for (Preprocessing_Algorithm p_alg : fas) {
						FVSHelper.getInstance().logFile("Preprocessing: " + p_alg);
						pt = FVSHelper.getInstance().getPreprocessType(p_alg);
						FVSHelper.getInstance().logFile("Preprocess type: " + pt);
						if (p_alg == Preprocessing_Algorithm.Original) {
							runEvaluation(cts, datasetName, dis_alg, discretized, p_alg, "Original", -999,
									ThresholdType.NA, null);
						} else if (pt == PreprocessingType.FVS) {
							if (p_alg == Preprocessing_Algorithm.FVS_Random || p_alg == Preprocessing_Algorithm.FVS_Random_Entropy) {
								for (int i = 0; i < DOUBLE_PARAMS.length; i++) {
									Double double_param = DOUBLE_PARAMS[i];
									if (double_param == 1 && p_alg == Preprocessing_Algorithm.FVS_Random)
										continue;
									else if (double_param == 0)
										continue;
									Filter filter = getFVS(p_alg, ThresholdType.Iteration, discretized.numInstances(),
											double_param);
									runEvaluation(cts, datasetName, dis_alg, discretized, p_alg,
											(p_alg + " : " + double_param), double_param, ThresholdType.Iteration,
											filter);
									filter = null;
								}
							} else if (p_alg == Preprocessing_Algorithm.FVS_Entropy) {
								for (ThresholdType thr_alg : tts) {
									if (thr_alg == ThresholdType.NA)
										continue;
									if (thr_alg == ThresholdType.Iteration) {
										for (int i = 0; i < DOUBLE_PARAMS.length; i++) {
											Double double_param = DOUBLE_PARAMS[i];
											if ((double_param == 1 || double_param == 0))
												continue;
											Filter filter = getFVS(p_alg, ThresholdType.Iteration,
													discretized.numInstances(), double_param);
											runEvaluation(cts, datasetName, dis_alg, discretized, p_alg,
													("FVS Entropy Iteration : " + double_param), double_param,
													ThresholdType.Iteration, filter);
											filter = null;
										}
									} else {
										Filter filter = getFVS(p_alg, thr_alg, discretized.numInstances(), 0.0);
										runEvaluation(cts, datasetName, dis_alg, discretized, p_alg, p_alg.toString(),
												0.0, thr_alg, filter);
										filter = null;
									}
								}
							} else if (p_alg == Preprocessing_Algorithm.FVS_Correlation) {
								for (ThresholdType thr_alg : tts) {
									if (thr_alg == ThresholdType.NA || thr_alg == ThresholdType.Iteration)
										continue;
									for (int i = 0; i < DOUBLE_PARAMS.length; i++) {
										Double double_param = DOUBLE_PARAMS[i];
										if ((double_param == 1 || double_param == 0))
											continue;
										Filter filter = getFVS(p_alg, thr_alg, discretized.numInstances(),
												double_param);
										runEvaluation(cts, datasetName, dis_alg,
												discretized, p_alg, String.format("FVS Correlation (%s) : %.1f ",
														thr_alg.toString(), double_param),
												double_param, thr_alg, filter);
										filter = null;
									}
								}
							}
						} else if (pt != PreprocessingType.None) {
							Filter filter = getFilter(discretized, p_alg);
							runEvaluation(cts, datasetName, dis_alg, discretized, p_alg, p_alg.toString(), 0.0,
									ThresholdType.NA, filter);
							filter = null;
						}
					} // For each preprocessing algorithm
					System.out.println();
					if (discretized != null)
						discretized.delete();
					discretized = null;
					System.gc();
				} // For each discretization
			} // Try for each file
			catch (Exception e) {
				if (FVSHelper.getInstance().getDebugStatus())
					e.printStackTrace();
				FVSHelper.getInstance().logFile(e.getMessage());
			}
		} // For each file

		System.out.println("Program finished");
		System.out.println(new Date());

	}

	private void runEvaluation(List<ClassifierType> cts, String datasetName, DiscretizationType dis_alg,
			Instances instances, Preprocessing_Algorithm p_alg, String context, double double_param,
			ThresholdType thr_alg, Filter filter) throws Exception, IOException {
		for (ClassifierType type : cts) {
			if (type == ClassifierType.DecisionStump && p_alg != Preprocessing_Algorithm.Original)
				continue;
			FVSHelper.getInstance().logFile("Classifier: " + type);
			double modelSize = Double.NEGATIVE_INFINITY;
			if (modelSize == Double.NEGATIVE_INFINITY) {
				FVSEvaluation eval = new FVSEvaluation(instances);
				// Cross validate dataset
				eval.stratifiedFold(type, CROSS_VALIDATION, p_alg, filter);
				if (p_alg == Preprocessing_Algorithm.FVS_Entropy)
					eval.setDouble_param(((EntropyFVS) ((FVS_Filter) filter).getFvs()).getThreshold());
				else
					eval.setDouble_param(double_param);

				writeReport(REPORT_FOLDER, datasetName, instances.classIndex(), eval, eval.getDouble_param(), p_alg,
						type, dis_alg, thr_alg);
				if (FVSHelper.getInstance().getDebugStatus())
					System.out.println("Writing report: " + context);
				eval = null;
			}
		}
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

	private Instances loadData(String file) throws Exception {
		DataSource source = new DataSource(file);
		if (FVSHelper.getInstance().getDebugStatus()) {
			FVSHelper.getInstance().logFile(file);
			if (FVSHelper.getInstance().getDebugStatus())
				System.err.println(new Date().toString());
		}
		Instances data = source.getDataSet();
		if (data.classIndex() == -1)
			data.setClassIndex(data.numAttributes() - 1);
		return data;
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
			((weka.filters.unsupervised.attribute.Discretize) filter).setBins(NUMBER_OF_BINS);
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
		Filter filter = getFVS(algo, thr_alg, numInstances, params);
		try {
			if (filter == null)
				return data;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
			filter = null;
		} catch (Exception e) {
			e.printStackTrace();
			FVSHelper.getInstance().logFile(e.getMessage());
		}
		return result;
	}

	private FVS_Filter getFVS(Preprocessing_Algorithm algo, ThresholdType thr_alg, int numInstances, Double... params) {
		FVS_Filter filter = new FVS_Filter(algo, thr_alg, numInstances, params);
		return filter;
	}

	private Filter getFilter(Instances data, Preprocessing_Algorithm algo) {
		Filter filter = null;
		switch (algo) {
		case FS_CFS:
			filter = new AttributeSelection();
			AttributeSelection temp = (AttributeSelection) filter;
			CfsSubsetEval cfs = new CfsSubsetEval();
			temp.setEvaluator(cfs);
			break;
		case FS_Consistency:
			filter = new AttributeSelection();
			temp = (AttributeSelection) filter;
			ConsistencySubsetEval cs = new ConsistencySubsetEval();
			temp.setEvaluator(cs);
			break;
		case FT_RandomProjection:
			RandomProjection rp = new RandomProjection();
			rp.setNumberOfAttributes(data.numAttributes() / 2);
			filter = rp;
			break;
		case FT_PCA:
			PrincipalComponents pca = new PrincipalComponents();
			pca.setMaximumAttributes(data.numAttributes() / 2);
			filter = pca;
			break;
		case IS_Reservoir:
			ReservoirSample rs = new ReservoirSample();
			/* Sample 5% of the original instance */
			rs.setSampleSize(data.numInstances() / 20);
			filter = rs;
			break;
		case IS_Misclassified:
			RemoveMisclassified rmc = new RemoveMisclassified();
			rmc.setClassifier(new J48());
			rmc.setClassIndex(data.classIndex());
			filter = rmc;
			break;
		default:
			break;
		}
		return filter;
	}

	private Instances applySelection(Instances data, Preprocessing_Algorithm algo) {
		Instances result = null;
		Filter filter = getFilter(data, algo);
		try {
			if (filter == null)
				return data;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
			filter = null;
		} catch (Exception e) {
			e.printStackTrace();
			FVSHelper.getInstance().logFile(e.getMessage());
		}
		return result;
	}

	private void writeReport(String folder, String datasetName, int classIndex, FVSEvaluation eval, double double_param,
			Object... params) throws IOException {
		if (!folder.endsWith("/"))
			folder = folder + "/";
		if (datasetName.contains(".")) {
			datasetName = datasetName.split("\\.")[0];
		}
		String uid = datasetName;
		datasetName = datasetName + ".txt";
		File ffolder = new File(folder);
		ffolder.mkdirs();
		File f = new File(folder + datasetName);
		boolean new_file = f.createNewFile();
		FileWriter fileWriter = new FileWriter(f, true);
		if (new_file) {
			fileWriter.write(REPORT_HEADER);
		}

		int noise = 0;
		if (FVSHelper.getInstance().getAddNoise())
			noise = FVSHelper.getInstance().getNoiseLevel();
		String threshold = double_param + "";
		if (threshold.trim().equalsIgnoreCase("nan"))
			threshold = params[3].toString();
		String discretization = params[2].toString();
		if (params[2].toString() == DiscretizationType.Binning.toString()
				|| params[2].toString() == DiscretizationType.Frequency.toString())
			discretization = discretization + "_" + NUMBER_OF_BINS;
		fileWriter.append(String.format(REPORT_FORMAT, uid, params[0].toString(), params[1].toString(), discretization,
				params[3].toString(), eval.getAccuracy(), eval.getRuleSize(), double_param, eval.getModelSize(),
				eval.getRunTime(), eval.getMemoryUsage(), noise));
		fileWriter.close();
		fileWriter = null;
	}

}
