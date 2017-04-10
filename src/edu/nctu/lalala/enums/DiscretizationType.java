package edu.nctu.lalala.enums;

public enum DiscretizationType {
	None, Binning, Frequency,
	/**
	 * Fayyad, Usama, and Keki Irani. "Multi-interval discretization of
	 * continuous-valued attributes for classification learning." (1993).
	 */
	MDL,
	/**
	 * Yang, Ying, and Geoffrey I. Webb. "Proportional k-interval discretization
	 * for naive-Bayes classifiers." European Conference on Machine Learning.
	 * Springer Berlin Heidelberg, 2001.
	 */
	PKI
}
