package de.edux.ml.cnn.tensor;

import java.util.Arrays;
import java.io.Serializable;

public abstract class Tensor<T extends Number> implements Serializable {
    protected T[] data;
    protected int[] shape;
    protected int[] strides;
    protected boolean requiresGrad;
    protected Tensor<T> grad;
    
    public Tensor(int[] shape, boolean requiresGrad) {
        this.shape = shape.clone();
        this.requiresGrad = requiresGrad;
        this.strides = computeStrides(shape);
        initializeData();
    }
    
    protected abstract void initializeData();
    protected abstract T zero();
    protected abstract T add(T a, T b);
    protected abstract T multiply(T a, T b);
    protected abstract T divide(T a, T b);
    protected abstract T subtract(T a, T b);
    
    private int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }
    
    public int[] getShape() {
        return shape.clone();  
    }
    
    public int size() {
        return Arrays.stream(shape).reduce(1, (a, b) -> a * b);
    }
    
    public T get(int... indices) {
        return data[getFlatIndex(indices)];
    }
    
    public void set(T value, int... indices) {
        data[getFlatIndex(indices)] = value;
    }
    
    private int getFlatIndex(int... indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Index dimensions don't match tensor dimensions");
        }
        int flatIndex = 0;
        for (int i = 0; i < indices.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index out of bounds");
            }
            flatIndex += indices[i] * strides[i];
        }
        return flatIndex;
    }
    
    public abstract Tensor<T> add(Tensor<T> other);
    public abstract Tensor<T> multiply(Tensor<T> other);
    public abstract Tensor<T> matmul(Tensor<T> other);
    public abstract Tensor<T> reshape(int... newShape);
    public abstract Tensor<T> transpose();
    
    public boolean requiresGrad() {
        return requiresGrad;
    }
    
    public Tensor<T> getGrad() {
        return grad;
    }
    
    public void setGrad(Tensor<T> grad) {
        this.grad = grad;
    }
    
    public T[] getData() {
        return data.clone();
    }
}