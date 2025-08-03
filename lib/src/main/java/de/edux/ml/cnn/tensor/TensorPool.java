package de.edux.ml.cnn.tensor;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.Arrays;

public class TensorPool {
    private static final ConcurrentHashMap<String, ConcurrentLinkedQueue<FloatTensor>> tensorPools = new ConcurrentHashMap<>();
    private static final int MAX_POOL_SIZE = 500;
    
    private static String shapeKey(int[] shape) {
        return Arrays.toString(shape);
    }
    
    public static FloatTensor get(int[] shape) {
        String key = shapeKey(shape);
        ConcurrentLinkedQueue<FloatTensor> pool = tensorPools.get(key);
        
        if (pool != null) {
            FloatTensor tensor = pool.poll();
            if (tensor != null) {
                float[] data = tensor.getPrimitiveData();
                Arrays.fill(data, 0.0f);
                tensor.syncFromPrimitive();
                return tensor;
            }
        }
        
        return new FloatTensor(shape);
    }
    
    public static void release(FloatTensor tensor) {
        if (tensor == null) return;
        
        String key = shapeKey(tensor.getShape());
        ConcurrentLinkedQueue<FloatTensor> pool = tensorPools.computeIfAbsent(key, k -> new ConcurrentLinkedQueue<>());
        
        if (pool.size() < MAX_POOL_SIZE) {
            pool.offer(tensor);
        } else {
            tensor.dispose();
        }
    }
    
    public static void clear() {
        for (ConcurrentLinkedQueue<FloatTensor> pool : tensorPools.values()) {
            FloatTensor tensor;
            while ((tensor = pool.poll()) != null) {
                tensor.dispose();
            }
        }
        tensorPools.clear();
    }
    
    public static int getPoolSize(int[] shape) {
        String key = shapeKey(shape);
        ConcurrentLinkedQueue<FloatTensor> pool = tensorPools.get(key);
        return pool != null ? pool.size() : 0;
    }
}