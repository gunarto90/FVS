package edu.nctu.lalala.util;

public class FVSHelper {
	private FVSHelper() {
	}
	
	private static final FVSHelper singleton = new FVSHelper();
	
	public static FVSHelper getInstance() {
		return singleton;
	}

}
