package edu.nctu.lalala.fvs.evaluation;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.Random;

import edu.nctu.lalala.enums.ClassifierType;
import edu.nctu.lalala.enums.Preprocessing_Algorithm;
import edu.nctu.lalala.util.FVSHelper;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.ConsistencySubsetEval;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RandomProjection;
import weka.filters.unsupervised.instance.RemoveMisclassified;
import weka.filters.unsupervised.instance.ReservoirSample;

public class FVSEvaluation extends weka.classifiers.Evaluation {
	String format;
	String header;
	int numOfBins;
	Instances data;
	double accuracy;
	double modelSize;
	int ruleSize;

	public FVSEvaluation(Instances data, String format, String header, int numOfBins) throws Exception {
		super(data);
		this.data = data;
		this.format = format;
		this.header = header;
		this.numOfBins = numOfBins;
		this.accuracy = 0.0;
		this.modelSize = 0.0;
		this.ruleSize = 0;
	}

	/**
	 * 
	 */
	private static final long serialVersionUID = -234672101871792496L;

	public void stratifiedFold(ClassifierType type, int folds) {
		stratifiedFold(type, folds, null);
	}

	public void stratifiedFold(ClassifierType type, int folds, Filter filter) {
		int seed = 10000;
		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);
		randData.stratify(folds);
		System.err.println("Stratified kfold: " + folds);

		double[] accuracies = new double[folds];
		double[] models = new double[folds];
		int[] rules = new int[folds];

		/* To get the base model size -- weka dump */
		double baseModel = 0.0;
		Classifier tempClassifier = null;
		try {
			tempClassifier = buildClassifier(new Instances(data, 0), type);
			double[] tempModel = getModelSize(tempClassifier);
			baseModel = tempModel[0];
			tempModel = null;
		} catch (Exception e1) {
		}
		tempClassifier = null;

		for (int n = 0; n < folds; n++) {
			Instances train = randData.trainCV(folds, n);
			Instances test = randData.testCV(folds, n);

			if (filter != null) {

			}

			int correct = 0;
			double accuracy = 0.0;
			double model = 0.0;
			int rule = 0;
			double pred, label;

			try {
				Classifier cl = buildClassifier(data, type);
				for (int i = 0; i < test.numInstances(); i++) {
					pred = cl.classifyInstance(test.instance(i));
					label = test.instance(i).classValue();
					if (pred == label)
						correct++;
				}
				accuracy = (double) correct / test.numInstances();
				accuracies[n] = accuracy;
				double[] tempModel = getModelSize(cl);
				models[n] = tempModel[0] - baseModel;
				rules[n] = (int) tempModel[1];
				tempModel = null;

			} catch (Exception e) {

			}
		}

		/* Calculate metrics */
		for (double x : accuracies) {
			this.accuracy += x;
		}
		this.accuracy /= accuracies.length;
		for (double x : models) {
			this.modelSize += x;
		}
		this.modelSize /= models.length;
		for (int x : rules) {
			this.ruleSize += x;
		}
		this.ruleSize /= rules.length;

		/* Clearing the memory */
		accuracies = null;
		models = null;
		rules = null;

		System.gc();

		System.err.println(this.accuracy);
	}

	public double getAccuracy() {
		return this.accuracy;
	}

	public double getModelSize() {
		return this.modelSize;
	}

	public int getRuleSize() {
		return this.ruleSize;
	}

	private double[] getModelSize(Classifier cl) {
		double[] result = new double[2];
		double modelSize = 0.0;
		double rule = 0.0; // Only for J48 and Jrip
		try {
			J48 a = (J48) cl;
			rule = a.measureNumRules();
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
		} catch (IOException e) {

		}
		result[0] = modelSize;
		result[1] = rule;
		return result;
	}

	private Instances applySelection(Instances data, Preprocessing_Algorithm algo) {
		Instances result = null;
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
			filter = rp;
			break;
		case FT_PCA:
			PrincipalComponents pca = new PrincipalComponents();
			pca.setMaximumAttributes(data.numAttributes());
			filter = pca;
			break;
		case IS_Reservoir:
			ReservoirSample rs = new ReservoirSample();
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
		try {
			if (filter == null)
				return data;
			filter.setInputFormat(data);
			result = Filter.useFilter(data, filter);
		} catch (Exception e) {
			FVSHelper.getInstance().logFile(e.getMessage());
		}
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
		default:
			c = new J48();
			((J48) c).setUnpruned(true);
			break;
		}
		c.buildClassifier(data);
		return c;
	}

}
