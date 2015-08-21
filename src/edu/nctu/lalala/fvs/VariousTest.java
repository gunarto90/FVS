package edu.nctu.lalala.fvs;

import java.io.File;
import java.io.IOException;

import org.junit.Before;
import org.junit.Test;

public class VariousTest {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void test() {
		try {
			File f_ori = File.createTempFile("weka", "model");
			System.out.println(f_ori.toString());
			System.out.println(f_ori.getAbsolutePath());
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
