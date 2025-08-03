package de.edux.ml.cnn.tensor;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.io.Serializable;

public class FloatTensor extends Tensor<Float> implements Serializable {
    private float[] primitiveData;
    private static final transient ConcurrentHashMap<Integer, ConcurrentLinkedQueue<float[]>> memoryPool = new ConcurrentHashMap<>();
    private static final int MAX_POOL_SIZE = 1000;
    
    public FloatTensor(int[] shape, boolean requiresGrad) {
        super(shape, requiresGrad);
    }
    
    public FloatTensor(int[] shape) {
        this(shape, false);
    }
    
    private float[] getPooledArray(int size) {
        ConcurrentLinkedQueue<float[]> pool = memoryPool.get(size);
        if (pool != null) {
            float[] array = pool.poll();
            if (array != null) {
                Arrays.fill(array, 0.0f);
                return array;
            }
        }
        return new float[size];
    }
    
    private void returnToPool(float[] array) {
        if (array == null) return;
        int size = array.length;
        ConcurrentLinkedQueue<float[]> pool = memoryPool.computeIfAbsent(size, k -> new ConcurrentLinkedQueue<>());
        if (pool.size() < MAX_POOL_SIZE) {
            pool.offer(array);
        }
    }
    
    public static FloatTensor zeros(int... shape) {
        return new FloatTensor(shape);
    }
    
    public static FloatTensor ones(int... shape) {
        FloatTensor tensor = new FloatTensor(shape);
        Arrays.fill(tensor.primitiveData, 1.0f);
        Arrays.fill(tensor.data, 1.0f);
        return tensor;
    }
    
    public static FloatTensor fromArray(float[] data, int... shape) {
        int totalSize = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
        if (data.length != totalSize) {
            throw new IllegalArgumentException("Data length doesn't match tensor size");
        }
        
        FloatTensor tensor = new FloatTensor(shape);
        System.arraycopy(data, 0, tensor.primitiveData, 0, data.length);
        for (int i = 0; i < data.length; i++) {
            tensor.data[i] = data[i];
        }
        return tensor;
    }
    
    @Override
    protected void initializeData() {
        int totalSize = size();
        primitiveData = getPooledArray(totalSize);
        data = new Float[totalSize];
        for (int i = 0; i < totalSize; i++) {
            data[i] = 0.0f;
        }
    }
    
    public void dispose() {
        returnToPool(primitiveData);
        primitiveData = null;
    }
    
    @Override
    protected Float zero() {
        return 0.0f;
    }
    
    @Override
    protected Float add(Float a, Float b) {
        return a + b;
    }
    
    @Override
    protected Float multiply(Float a, Float b) {
        return a * b;
    }
    
    @Override
    protected Float divide(Float a, Float b) {
        return a / b;
    }
    
    @Override
    protected Float subtract(Float a, Float b) {
        return a - b;
    }
    
    @Override
    public FloatTensor add(Tensor<Float> other) {
        if (!Arrays.equals(this.shape, other.getShape())) {
            throw new IllegalArgumentException("Tensor shapes must match for addition");
        }
        
        FloatTensor result = TensorPool.get(this.shape);
        FloatTensor otherFloat = (FloatTensor) other;
        
        for (int i = 0; i < this.primitiveData.length; i++) {
            result.primitiveData[i] = this.primitiveData[i] + otherFloat.primitiveData[i];
            result.data[i] = result.primitiveData[i];
        }
        return result;
    }
    
    public void addInPlace(FloatTensor other) {
        if (!Arrays.equals(this.shape, other.getShape())) {
            throw new IllegalArgumentException("Tensor shapes must match for addition");
        }
        
        for (int i = 0; i < this.primitiveData.length; i++) {
            this.primitiveData[i] += other.primitiveData[i];
            this.data[i] = this.primitiveData[i];
        }
    }
    
    @Override
    public FloatTensor multiply(Tensor<Float> other) {
        if (!Arrays.equals(this.shape, other.getShape())) {
            throw new IllegalArgumentException("Tensor shapes must match for multiplication");
        }
        
        FloatTensor result = TensorPool.get(this.shape);
        FloatTensor otherFloat = (FloatTensor) other;
        
        for (int i = 0; i < this.primitiveData.length; i++) {
            result.primitiveData[i] = this.primitiveData[i] * otherFloat.primitiveData[i];
            result.data[i] = result.primitiveData[i];
        }
        return result;
    }
    
    @Override
    public FloatTensor matmul(Tensor<Float> other) {
        if (this.shape.length != 2 || other.getShape().length != 2) {
            throw new IllegalArgumentException("Matrix multiplication requires 2D tensors");
        }
        
        int[] otherShape = other.getShape();
        if (this.shape[1] != otherShape[0]) {
            throw new IllegalArgumentException("Matrix dimensions don't match for multiplication");
        }
        
        int[] resultShape = new int[]{this.shape[0], otherShape[1]};
        FloatTensor result = TensorPool.get(resultShape);
        FloatTensor otherFloat = (FloatTensor) other;
        
        int M = this.shape[0];
        int K = this.shape[1];
        int N = otherShape[1];
        
        int blockSize = Math.min(64, Math.max(8, (int) Math.sqrt(K)));
        
        for (int ii = 0; ii < M; ii += blockSize) {
            for (int jj = 0; jj < N; jj += blockSize) {
                for (int kk = 0; kk < K; kk += blockSize) {
                    int iMax = Math.min(ii + blockSize, M);
                    int jMax = Math.min(jj + blockSize, N);
                    int kMax = Math.min(kk + blockSize, K);
                    
                    for (int i = ii; i < iMax; i++) {
                        for (int j = jj; j < jMax; j++) {
                            float sum = result.primitiveData[i * N + j];
                            for (int k = kk; k < kMax; k++) {
                                sum += this.primitiveData[i * K + k] * otherFloat.primitiveData[k * N + j];
                            }
                            result.primitiveData[i * N + j] = sum;
                        }
                    }
                }
            }
        }
        
        for (int i = 0; i < result.primitiveData.length; i++) {
            result.data[i] = result.primitiveData[i];
        }
        
        return result;
    }
    
    @Override
    public FloatTensor reshape(int... newShape) {
        int newSize = Arrays.stream(newShape).reduce(1, (a, b) -> a * b);
        if (newSize != this.size()) {
            throw new IllegalArgumentException("New shape must have same total size");
        }
        
        FloatTensor result = TensorPool.get(newShape);
        System.arraycopy(this.primitiveData, 0, result.primitiveData, 0, this.primitiveData.length);
        System.arraycopy(this.data, 0, result.data, 0, this.data.length);
        return result;
    }
    
    @Override
    public FloatTensor transpose() {
        if (this.shape.length != 2) {
            throw new IllegalArgumentException("Transpose only supported for 2D tensors");
        }
        
        int[] newShape = new int[]{this.shape[1], this.shape[0]};
        FloatTensor result = TensorPool.get(newShape);
        
        int rows = this.shape[0];
        int cols = this.shape[1];
        int blockSize = 32;
        
        for (int ii = 0; ii < rows; ii += blockSize) {
            for (int jj = 0; jj < cols; jj += blockSize) {
                int iMax = Math.min(ii + blockSize, rows);
                int jMax = Math.min(jj + blockSize, cols);
                
                for (int i = ii; i < iMax; i++) {
                    for (int j = jj; j < jMax; j++) {
                        result.primitiveData[j * rows + i] = this.primitiveData[i * cols + j];
                        result.data[j * rows + i] = this.data[i * cols + j];
                    }
                }
            }
        }
        
        return result;
    }
    
    public FloatTensor sum(int axis) {
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Invalid axis");
        }
        
        int[] newShape = new int[shape.length - 1];
        int newIdx = 0;
        for (int i = 0; i < shape.length; i++) {
            if (i != axis) {
                newShape[newIdx++] = shape[i];
            }
        }
        
        if (newShape.length == 0) {
            newShape = new int[]{1};
        }
        
        FloatTensor result = TensorPool.get(newShape);
        
        int outerSize = 1;
        for (int i = 0; i < axis; i++) {
            outerSize *= shape[i];
        }
        
        int innerSize = 1;
        for (int i = axis + 1; i < shape.length; i++) {
            innerSize *= shape[i];
        }
        
        for (int outer = 0; outer < outerSize; outer++) {
            for (int inner = 0; inner < innerSize; inner++) {
                float sum = 0.0f;
                for (int axisIdx = 0; axisIdx < shape[axis]; axisIdx++) {
                    int srcIdx = outer * shape[axis] * innerSize + axisIdx * innerSize + inner;
                    sum += primitiveData[srcIdx];
                }
                int dstIdx = outer * innerSize + inner;
                if (dstIdx < result.primitiveData.length) {
                    result.primitiveData[dstIdx] = sum;
                    result.data[dstIdx] = sum;
                }
            }
        }
        
        return result;
    }
    
    public Float[] getDataArray() {
        return data;
    }
    
    public float[] getDataArrayPrimitive() {
        return primitiveData.clone();
    }
    
    public float[] getPrimitiveData() {
        return primitiveData;
    }
    
    public void setPrimitiveData(float[] data) {
        System.arraycopy(data, 0, this.primitiveData, 0, Math.min(data.length, this.primitiveData.length));
        for (int i = 0; i < this.primitiveData.length; i++) {
            this.data[i] = this.primitiveData[i];
        }
    }
    
    public void syncFromPrimitive() {
        for (int i = 0; i < this.primitiveData.length; i++) {
            this.data[i] = this.primitiveData[i];
        }
    }
    
    public void syncToPrimitive() {
        for (int i = 0; i < this.data.length; i++) {
            this.primitiveData[i] = this.data[i];
        }
    }
}