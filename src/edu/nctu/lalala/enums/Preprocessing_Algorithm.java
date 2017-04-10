package edu.nctu.lalala.enums;

public enum Preprocessing_Algorithm {
	/* Feature Value Selection methods */
	/**
	 * Original data
	 */
	Original,
	/**
	 * Randomized FVS
	 */
	FVS_Random,
	/**
	 * Entropy-based FVS
	 */
	FVS_Entropy,
	/**
	 * Correlation-based FVS
	 */
	FVS_Correlation,
	/* Instance Selection methods */
	/**
	 * Vitter, Jeffrey S. "Random sampling with a reservoir." ACM Transactions
	 * on Mathematical Software (TOMS) 11.1 (1985): 37-57.
	 */
	IS_Reservoir,
	/**
	 * A filter that removes instances which are incorrectly classified. Useful
	 * for removing outliers.
	 */
	IS_Misclassified,
	/* Feature Selection methods */
	/**
	 * Hall, Mark A. Correlation-based feature selection for machine learning.
	 * Diss. The University of Waikato, 1999.
	 */
	FS_CFS,
	/**
	 * Liu, Huan, and Rudy Setiono. "A probabilistic approach to feature
	 * selection-a filter solution." ICML. Vol. 96. 1996.
	 */
	FS_Consistency,
	/**
	 * Fradkin, Dmitriy, and David Madigan. "Experiments with random projections
	 * for machine learning." Proceedings of the ninth ACM SIGKDD international
	 * conference on Knowledge discovery and data mining. ACM, 2003.
	 */
	FS_RandomProjection,
	/**
	 * Bennett, K. P., and M. J. Embrechts. "An optimization perspective on
	 * kernel partial least squares regression." Nato Science Series sub series
	 * III computer and systems sciences 190 (2003): 227-250.
	 */
	FS_Kernel,
	/**
	 * Evaluates the worth of an attribute by measuring the correlation
	 * (Pearson's) between it and the class.
	 */
	FS_CorrAttr,
	/**
	 * Evaluates the worth of an attribute by measuring the gain ratio with
	 * respect to the class. InfoGain(Class,Attribute) = H(Class) - H(Class |
	 * Attribute).
	 */
	FS_GainRatio,
	/**
	 * Evaluates the worth of an attribute by measuring the symmetrical
	 * uncertainty with respect to the class. SymmU(Class, Attribute) = 2 *
	 * (H(Class) - H(Class | Attribute)) / H(Class) + H(Attribute).
	 */
	FS_SymmetricUncertainty,
	/**
	 * Performs a greedy forward or backward search through the space of
	 * attribute subsets. May start with no/all attributes or from an arbitrary
	 * point in the space. Stops when the addition/deletion of any remaining
	 * attributes results in a decrease in evaluation. Can also produce a ranked
	 * list of attributes by traversing the space from one side to the other and
	 * recording the order that attributes are selected.
	 */
	FS_GreedyStepwise,
	/**
	 * Marko Robnik-Sikonja, Igor Kononenko: An adaptation of Relief for
	 * attribute estimation in regression. In: Fourteenth International
	 * Conference on Machine Learning, 296-304, 1997.
	 */
	FS_Relief,
	/**
	 * Kohavi, Ron, and George H. John. "Wrappers for feature subset selection."
	 * Artificial intelligence 97.1-2 (1997): 273-324.
	 */
	FS_Wrapper
}
