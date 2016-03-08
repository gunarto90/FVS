package edu.nctu.lalala.fvs;

import java.util.ArrayList;
import java.util.List;

public class CorrelationMatrix {
	private List<Double> corrValues;
	private Double[][] CM;
	
	public CorrelationMatrix()
	{
		corrValues = new ArrayList<Double>();
	}

	public List<Double> getCorrValues() {
		return corrValues;
	}

	public Double[][] getCM() {
		return CM;
	}

	public void setCM(Double[][] cM) {
		CM = cM;
	}
}