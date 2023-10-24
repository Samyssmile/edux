package de.edux.math.entity;

import de.edux.math.EntityTest;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VectorTest implements EntityTest {

    static Vector first;
    static Vector second;

    @BeforeAll
    public static void init() {
        first = new Vector(new double[] {1, 5, 4});
        second = new Vector(new double[] {3, 8, 0});
    }

    @Test
    @Override
    public void testAdd() {
        assertEquals(new Vector(new double[] {4, 13, 4}), first.add(second));
    }

    @Test
    @Override
    public void testSubtract() {
        assertEquals(new Vector(new double[] {-2, -3, 4}), first.subtract(second));
    }

    @Test
    @Override
    public void testMultiply() {
        assertEquals(new Vector(new double[] {3, 40, 0}), first.multiply(second));
    }

    @Test
    @Override
    public void testScalarMultiply() {
        assertEquals(new Vector(new double[] {3, 15, 12}), first.scalarMultiply(3)); // first by 3
        assertEquals(new Vector(new double[] {-6, -16, 0}), second.scalarMultiply(-2)); // second by -2
    }

    @Test
    public void testDot() {
        assertEquals(43, first.dot(second));
    }

}
