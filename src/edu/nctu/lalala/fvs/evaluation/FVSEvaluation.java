package edu.nctu.lalala.fvs.evaluation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import edu.nctu.lalala.enums.ClassifierType;
import edu.nctu.lalala.enums.PreprocessingType;
import edu.nctu.lalala.enums.Preprocessing_Algorithm;
import edu.nctu.lalala.util.FVSHelper;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.neighboursearch.KDTree;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddNoise;

public class FVSEvaluation extends weka.classifiers.Evaluation {
	int numOfBins;
	Instances data;
	double accuracy;
	long modelSize;
	long ruleSize;
	double runTime;
	double memoryUsage;

	public FVSEvaluation(Instances data, int numOfBins) throws Exception {
		super(data);
		this.data = data;
		this.numOfBins = numOfBins;
		this.accuracy = 0.0;
		this.modelSize = 0;
		this.ruleSize = 0;
		this.runTime = 0;
		this.memoryUsage = 0;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -234672101871792496L;

	public void stratifiedFold(ClassifierType type, int folds, Preprocessing_Algorithm p_alg) {
		stratifiedFold(type, folds, p_alg, null);
	}

	public void stratifiedFold(ClassifierType type, int folds, Preprocessing_Algorithm p_alg, Filter filter) {
		int seed = 10000;
		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		randData.stratify(folds);
		// System.err.println("Stratified kfold: " + folds);

		double[] accuracies = new double[folds];
		long[] models = new long[folds];
		long[] rules = new long[folds];
		double[] run_time = new double[folds];
		double[] memories = new double[folds];

		/* To get the base model size -- weka dump */
		long baseModel = 0;
		Classifier tempClassifier = null;
		try {
			tempClassifier = buildClassifier(new Instances(data, 0), type);
			long[] tempModel = getModelSize(tempClassifier);
			baseModel = tempModel[0];
			tempModel = null;
			// System.err.println("Base model size: " + baseModel);
		} catch (Exception e1) {
		}
		tempClassifier = null;

		/* Adding noise to training data */
		if (FVSHelper.getInstance().getAddNoise()) {
			if (FVSHelper.getInstance().getDebugStatus())
				System.err.println("Adding noise to the dataset");
			weka.filters.unsupervised.attribute.AddNoise noise = new AddNoise();
			int percent = 10;
			int seedNoise = 999;
			boolean useMissing = false;
			for (int i = 0; i < randData.numAttributes() - 1; i++) {
				noise.addNoise(randData, seedNoise, percent, i, useMissing);
			}
			noise = null;
		}

		for (int n = 0; n < folds; n++) {
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);
			PreprocessingType pt = FVSHelper.getInstance().getPreprocessType(p_alg);

			if (filter != null) {
				try {
					filter.setInputFormat(train);
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				Object[] temp = applyFilter(filter, train, pt);
				train = (Instances) temp[0];
				run_time[n] = (double) temp[1];
				memories[n] = (double) temp[2];
			}

			int correct = 0;
			double accuracy = 0.0;
			double pred, label;

			try {
				Classifier cl = buildClassifier(train, type);
				// Apply the same preprocessing to the test dataset
				if (filter != null) {
					Object[] temp = applyFilter(filter, test, pt);
					test = (Instances) temp[0];
					run_time[n] = (double) temp[1];
					memories[n] = (double) temp[2];
				}
				for (int i = 0; i < test.numInstances(); i++) {
					pred = cl.classifyInstance(test.instance(i));
					label = test.instance(i).classValue();
					if (pred == label)
						correct++;
				}
				accuracy = (double) correct / test.numInstances();
				accuracies[n] = accuracy;
				long[] tempModel = getModelSize(cl);
				models[n] = tempModel[0] - baseModel;
				rules[n] = tempModel[1];
				tempModel = null;

			} catch (Exception e) {

			}
		}

		/* Calculate metrics */
		for (double x : accuracies) {
			this.accuracy += x;
		}
		this.accuracy /= accuracies.length;
		for (long x : models) {
			this.modelSize += x;
		}
		this.modelSize /= models.length;
		for (long x : rules) {
			this.ruleSize += x;
		}
		this.ruleSize /= rules.length;
		if (this.runTime == 0.0) {
			for (double x : run_time) {
				this.runTime += x;
			}
			this.runTime /= run_time.length;
			this.runTime /= 1000000;
		}
		if (this.memoryUsage == 0.0) {
			for (double x : memories) {
				this.memoryUsage += x;
			}
			this.memoryUsage /= memories.length;
			this.memoryUsage /= (1024);
		}

		/* Clearing the memory */
		accuracies = null;
		models = null;
		rules = null;
		run_time = null;
		memories = null;

		System.gc();

		// System.err.println(this.accuracy);
	}

	public double getAccuracy() {
		return this.accuracy;
	}

	public long getModelSize() {
		return this.modelSize;
	}

	public long getRuleSize() {
		return this.ruleSize;
	}

	public double getRunTime() {
		return this.runTime;
	}

	public double getMemoryUsage() {
		return this.memoryUsage;
	}

	private long[] getModelSize(Classifier cl) {
		long[] result = new long[2];
		long modelSize, rule;
		modelSize = 0;
		rule = 0;
		try {
			J48 a = (J48) cl;
			rule = (int) a.measureNumRules();
		} catch (Exception e) {
		}
		if (rule == 0.0) {
			try {
				JRip a = (JRip) cl;
				rule = a.getRuleset().size();
			} catch (Exception e) {
			}
		}
		ObjectOutputStream oos;
		try {
			File f = File.createTempFile("weka", "model");
			oos = new ObjectOutputStream(new FileOutputStream(f));
			oos.writeObject(cl);
			oos.flush();
			oos.close();
			modelSize = f.length();
			f.deleteOnExit();
			f.delete();
		} catch (IOException e) {

		}
		result[0] = modelSize;
		result[1] = rule;

		// System.err.println("getModelSize: " + modelSize);
		return result;
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
		case Bayes:
			c = new NaiveBayes();
			break;
		case Logistic:
			c = new Logistic();
			break;
		case Instance:
			int k = 1;
			c = new IBk(k);
			((IBk) c).setNearestNeighbourSearchAlgorithm(new KDTree());
			break;
		default:
			c = new J48();
			((J48) c).setUnpruned(true);
			break;
		}
		c.buildClassifier(data);
		return c;
	}
	
	private Object[] applyFilter(Filter filter, Instances data, PreprocessingType pt)
	{
		Object[] results = new Object[3];
		try {
			double beforeMem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
					/ (1024 * 1024);
			double time = System.nanoTime();
			data = Filter.useFilter(data, filter);
			results[0] = data;
			time = (System.nanoTime() - time);
			results[1] = time;
			double afterMem = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
					/ (1024 * 1024);
			results[2] = afterMem - beforeMem;

		} catch (Exception e) {
			System.err.println("Error in applying filter in training data: " + pt);
			e.printStackTrace();
		}
		return results;
	}

}
