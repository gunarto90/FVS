package test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import edu.nctu.lalala.fvs.Value;

public class TestValue {

	@Before
	public void setUp() throws Exception {
	}

	@Test
	public void testEquality() {
		// Negative value hasn't been tested
		Value v1 = new Value("(2.48-6.31]");
		Value v2 = new Value("(6.31-inf)");
		Value v3 = new Value("(-inf-2.48]");
		Value v4 = new Value(-2.0);
		Value v5 = new Value(5.0);
		Value v6 = new Value(500.0);
//		System.out.println(v1);
//		System.out.println(v2);
//		System.out.println(v3);
//		System.out.println(v4);
//		System.out.println(v5);
//		System.out.println(v6);
		
		assertEquals((Value)v1, (Value)v5);
		assertEquals((Value)v3, (Value)v4);
		assertEquals((Value)v2, (Value)v6);
		assertNotEquals((Value)v1, (Value)v2);
		assertNotEquals((Value)v1, (Value)v3);
		assertNotEquals((Value)v1, (Value)v4);
		assertNotEquals((Value)v1, (Value)v6);
		assertNotEquals((Value)v2, (Value)v3);
		assertNotEquals((Value)v2, (Value)v4);
		assertNotEquals((Value)v2, (Value)v5);
		assertNotEquals((Value)v3, (Value)v5);
		assertNotEquals((Value)v3, (Value)v6);
	}
	
	@Test
	public void testList() {
		Value v1 = new Value("(2.48-6.31]");
		Value v2 = new Value("(6.31-inf)");
		Value v3 = new Value("(-inf-2.48]");
		Value v4 = new Value(-2.0);
		Value v5 = new Value(5.0);
		Value v6 = new Value(500.0);
		
		List<Value> list = new ArrayList<>();
		list.add(v1);
		list.add(v2);
		list.add(v3);
		
		Assert.assertTrue(list.contains(v4));
		Assert.assertTrue(list.contains(v5));
		Assert.assertTrue(list.contains(v6));
	}
}
