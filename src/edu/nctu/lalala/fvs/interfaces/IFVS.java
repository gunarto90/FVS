package edu.nctu.lalala.fvs.interfaces;

import weka.core.Instances;

public interface IFVS {
	public void input(Instances inst, Object[] params);
	public void applyFVS();
	public Instances output();
}
