package edu.nctu.lalala.fvs;

import java.util.Objects;

import edu.nctu.lalala.util.FVSHelper;

/**
 * Feature-Value pair class
 * 
 * @author Gunarto Sindoro Njoo
 * @version 1.0
 * @category Data Object
 *
 */
public class FV implements Comparable<FV> {
	/**
	 * Instance index number
	 */
	@SuppressWarnings("unused")
	private int index;
	/**
	 * Column index number
	 */
	private int feature;
	/**
	 * Value of the cell
	 */
	private Object value;
	/**
	 * Class label
	 */
	private double label;
	private double frequency;
	private double entropy;
	private double phi;
	private double ig;
	private double symmetricUncertainty;
	private double probability;
	private int numOfClassLabels;

	public FV(int feature, Object value, double label) {
		this.feature = feature;
		this.value = value;
		this.label = label;
		this.probability = 0.5;
	}

	@Override
	public int hashCode() {
		int hash = 5;
		hash = 71 * hash + this.getFeature();
		hash = 71 * hash + Objects.hashCode(this.getValue());
//		hash = (int) (71 * hash + this.getLabel());
		return hash;
	}

	@Override
	public boolean equals(Object obj) {
		if (obj == null) {
			return false;
		}
		if (getClass() != obj.getClass()) {
			return false;
		}
		final FV other = (FV) obj;
		if (this.getFeature() != other.getFeature()) {
			return false;
		}
		if (!Objects.equals(this.value, other.value)) {
			return false;
		}
//		if (!Objects.equals(this.label, other.label)) {
//			return false;
//		}
		return true;
	}

	@Override
	public String toString() {
//		String output = String.format("FV{%-3d : %15s (%2.0f) [E:%.3f] [F:%.3f]} ", this.feature, this.value, this.label, getEntropy(), getFrequency());
		String output = String.format("FV{%d:%s (%.0f) [E:%.3f] [F:%.3f] [IG:%.3f]}\n", this.feature, this.value, this.label, getEntropy(), getFrequency(), getIg());
//		String output = String.format("[%d:%s]", this.feature, this.value);
		return output;
	}
	
	@Override
	public int compareTo(FV arg0) {
		if (FVSHelper.getInstance().getInformationMetric().equals("ig"))
		{
			if(this.getIg() > arg0.getIg())
				return -1;
			else if(this.getIg() < arg0.getIg())
				return 1;
			else return 0;
		}
		else
		{
			if(this.getEntropy() < arg0.getEntropy())
				return -1;
			else if(this.getEntropy() > arg0.getEntropy())
				return 1;
			else return 0;
		}
		
	}

	public int getFeature() {
		return feature;
	}

	public void setFeature(int feature) {
		this.feature = feature;
	}

	public Object getValue() {
		return value;
	}

	public void setValue(Object value) {
		this.value = value;
	}

	public double getLabel() {
		return label;
	}

	public void setLabel(double label) {
		this.label = label;
	}

	public double getFrequency() {
		return frequency;
	}

	public void setFrequency(double frequency) {
		this.frequency = frequency;
	}

	public double getEntropy() {
		return entropy;
	}

	public void setEntropy(double entropy) {
		this.entropy = entropy;
	}

	public int getNumLabels() {
		return numOfClassLabels;
	}

	public void setNumLabels(int numOfClassLabels) {
		this.numOfClassLabels = numOfClassLabels;
	}

	public double getPhi() {
		return phi;
	}

	public void setPhi(double phi) {
		this.phi = phi;
	}

	public double getIg() {
		return ig;
	}

	public void setIg(double ig) {
		this.ig = ig;
	}

	public double getProbability() {
		return probability;
	}

	public void setProbability(double probability) {
		this.probability = probability;
	}

	public double getSymmetricUncertainty() {
		return symmetricUncertainty;
	}

	public void setSymmetricUncertainty(double symmetricUncertainty) {
		this.symmetricUncertainty = symmetricUncertainty;
	}
}
