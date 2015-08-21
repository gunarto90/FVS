package edu.nctu.lalala.fvs;

import java.util.Objects;

/**
 * Feature-Value pair class
 * 
 * @author Gunarto Sindoro Njoo
 * @version 1.0
 * @category Data Object
 *
 */
public class FV {
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
	private int frequency;
	private double entropy;

	public FV(int feature, Object value, double label) {
		this.feature = feature;
		this.value = value;
		this.label = label;
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
		String output = String.format("FV{%-3d : %12s (%2.0f) } ", this.feature, this.value, this.label);
		return output;
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

	public int getFrequency() {
		return frequency;
	}

	public void setFrequency(int frequency) {
		this.frequency = frequency;
	}

	public double getEntropy() {
		return entropy;
	}

	public void setEntropy(double entropy) {
		this.entropy = entropy;
	}
}
