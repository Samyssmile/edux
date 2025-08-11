package de.edux.ml.cnn.data;

import de.edux.ml.cnn.tensor.FloatTensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;
import java.lang.ref.SoftReference;

public class MNISTDataLoader implements DataLoader {
    private final int batchSize;
    private final List<DataSample> samples;
    private int currentIndex;
    private final Random random;
    private long lastProgressUpdate;
    
    private static class DataSample {
        final String imagePath;
        final int label;
        private SoftReference<FloatTensor> imageDataRef;
        private static final ConcurrentHashMap<String, SoftReference<FloatTensor>> imageCache = new ConcurrentHashMap<>();
        
        DataSample(String imagePath, int label, FloatTensor imageData) {
            this.imagePath = imagePath;
            this.label = label;
            this.imageDataRef = new SoftReference<>(imageData);
            // Cache the image data with soft reference for memory management
            imageCache.put(imagePath, new SoftReference<>(imageData));
        }
        
        FloatTensor getImageData() {
            FloatTensor data = imageDataRef.get();
            if (data == null) {
                // Try cache first
                SoftReference<FloatTensor> cachedRef = imageCache.get(imagePath);
                if (cachedRef != null) {
                    data = cachedRef.get();
                }
                if (data == null) {
                    // Reload from disk if not in cache
                    data = loadImageFromDisk(imagePath);
                    imageDataRef = new SoftReference<>(data);
                    imageCache.put(imagePath, new SoftReference<>(data));
                }
            }
            return data;
        }
        
        private static FloatTensor loadImageFromDisk(String imagePath) {
            try {
                BufferedImage image = ImageIO.read(new File(imagePath));
                return convertImageToTensor(image);
            } catch (IOException e) {
                throw new RuntimeException("Failed to reload image: " + imagePath, e);
            }
        }
    }
    
    public MNISTDataLoader(String csvPath, String datasetRoot, int batchSize) {
        this(csvPath, datasetRoot, batchSize, 1.0);
    }
    
    public MNISTDataLoader(String csvPath, String datasetRoot, int batchSize, double fraction) {
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.currentIndex = 0;
        this.random = new Random(42);
        this.lastProgressUpdate = 0;
        
        loadDataset(csvPath, datasetRoot, fraction);
    }
    
    private void loadDataset(String csvPath, String datasetRoot, double fraction) {
        try {
            System.out.println("Loading dataset from: " + csvPath);
            List<String> lines = Files.readAllLines(Paths.get(csvPath));
            int totalSamples = lines.size() - 1;
            int samplesToUse = (int) Math.ceil(totalSamples * fraction);
            
            System.out.printf("Using %.1f%% of dataset: %d out of %d samples%n", 
                             fraction * 100, samplesToUse, totalSamples);
            
            int progressUpdateInterval = Math.max(1, samplesToUse / 50);
            
            // Prepare data for parallel loading
            List<String[]> imagePaths = new ArrayList<>();
            for (int i = 1; i <= samplesToUse && i < lines.size(); i++) {
                String[] parts = lines.get(i).split(",");
                if (parts.length >= 2) {
                    imagePaths.add(new String[]{datasetRoot + "/" + parts[0], parts[1]});
                }
            }
            
            // Optimized parallel image loading with batch processing
            ConcurrentLinkedQueue<DataSample> loadedSamples = new ConcurrentLinkedQueue<>();
            int numThreads = Runtime.getRuntime().availableProcessors();
            int batchSize = Math.max(1, imagePaths.size() / (numThreads * 4));
            
            IntStream.range(0, imagePaths.size()).parallel().forEach(i -> {
                String[] pathData = imagePaths.get(i);
                String imagePath = pathData[0];
                int label = Integer.parseInt(pathData[1]);
                FloatTensor imageData = loadImageOptimized(imagePath);
                loadedSamples.add(new DataSample(imagePath, label, imageData));
                
                if ((i + 1) % progressUpdateInterval == 0 || (i + 1) == imagePaths.size()) {
                    printProgressBar(i + 1, imagePaths.size(), "Loading samples");
                }
            });
            
            // Convert to list and maintain order
            samples.addAll(loadedSamples);
            System.out.println();
            System.out.printf("Loaded %d samples successfully%n", samples.size());
        } catch (IOException e) {
            throw new RuntimeException("Failed to load dataset: " + e.getMessage(), e);
        }
    }
    
    private void printProgressBar(int current, int total, String operation) {
        long currentTime = System.currentTimeMillis();
        if (currentTime - lastProgressUpdate < 100 && current != total) {
            return;
        }
        lastProgressUpdate = currentTime;
        
        int width = 50;
        double progress = (double) current / total;
        int filled = (int) (width * progress);
        
        StringBuilder bar = new StringBuilder();
        bar.append("\r").append(operation).append(": [");
        
        for (int i = 0; i < width; i++) {
            if (i < filled) {
                bar.append("█");
            } else {
                bar.append("░");
            }
        }
        
        bar.append(String.format("] %.1f%% (%d/%d)", progress * 100, current, total));
        System.out.print(bar.toString());
        
        if (current == total) {
            System.out.println();
        }
    }
    
    private FloatTensor loadImageOptimized(String imagePath) {
        try {
            BufferedImage image = ImageIO.read(new File(imagePath));
            return convertImageToTensor(image);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load image: " + imagePath, e);
        }
    }
    
    private static FloatTensor convertImageToTensor(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        
        // Optimized pixel reading using getRGB array
        int[] rgbArray = new int[width * height];
        image.getRGB(0, 0, width, height, rgbArray, 0, width);
        
        // Check if image is grayscale by examining if R==G==B for all pixels
        boolean isGrayscale = true;
        for (int i = 0; i < rgbArray.length && isGrayscale; i++) {
            int rgb = rgbArray[i];
            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;
            if (r != g || g != b) {
                isGrayscale = false;
            }
        }
        
        if (isGrayscale) {
            // Single channel grayscale
            float[] imageData = new float[height * width];
            for (int i = 0; i < rgbArray.length; i++) {
                int rgb = rgbArray[i];
                int gray = (rgb >> 16) & 0xFF; // Use red channel (all channels are equal)
                imageData[i] = gray / 255.0f;
            }
            return FloatTensor.fromArray(imageData, 1, height, width);
        } else {
            // RGB image - 3 channels
            float[] imageData = new float[3 * height * width];
            for (int i = 0; i < rgbArray.length; i++) {
                int rgb = rgbArray[i];
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                
                // Channel-interleaved to channel-separated layout
                imageData[i] = r / 255.0f;                           // Red channel
                imageData[i + width * height] = g / 255.0f;          // Green channel  
                imageData[i + 2 * width * height] = b / 255.0f;      // Blue channel
            }
            return FloatTensor.fromArray(imageData, 3, height, width);
        }
    }
    
    private FloatTensor createOneHotLabel(int label, int numClasses) {
        float[] labelData = new float[numClasses];
        labelData[label] = 1.0f;
        return FloatTensor.fromArray(labelData, numClasses);
    }
    
    @Override
    public boolean hasNext() {
        return currentIndex < samples.size();
    }
    
    @Override
    public Batch next() {
        if (!hasNext()) {
            throw new NoSuchElementException("No more batches");
        }
        
        int actualBatchSize = Math.min(batchSize, samples.size() - currentIndex);
        List<FloatTensor> batchImages = new ArrayList<>();
        List<FloatTensor> batchLabels = new ArrayList<>();
        
        int currentBatch = (currentIndex / batchSize) + 1;
        int totalBatches = size();
        
        for (int i = 0; i < actualBatchSize; i++) {
            DataSample sample = samples.get(currentIndex + i);
            FloatTensor label = createOneHotLabel(sample.label, 10);
            
            batchImages.add(sample.getImageData());
            batchLabels.add(label);
        }
        
        currentIndex += actualBatchSize;
        
        FloatTensor batchData = stackTensors(batchImages);
        FloatTensor batchLabelData = stackTensors(batchLabels);
        
        return new Batch(batchData, batchLabelData);
    }
    
    private FloatTensor stackTensors(List<FloatTensor> tensors) {
        if (tensors.isEmpty()) {
            throw new IllegalArgumentException("Cannot stack empty tensor list");
        }
        
        int[] firstShape = tensors.get(0).getShape();
        int[] stackedShape = new int[firstShape.length + 1];
        stackedShape[0] = tensors.size();
        System.arraycopy(firstShape, 0, stackedShape, 1, firstShape.length);
        
        int totalSize = tensors.size() * tensors.get(0).size();
        float[] stackedData = new float[totalSize];
        
        // Optimized stacking using primitive arrays
        for (int i = 0; i < tensors.size(); i++) {
            float[] tensorData = tensors.get(i).getPrimitiveData();
            System.arraycopy(tensorData, 0, stackedData, i * tensorData.length, tensorData.length);
        }
        
        return FloatTensor.fromArray(stackedData, stackedShape);
    }
    
    @Override
    public void shuffle() {
        Collections.shuffle(samples, random);
    }
    
    @Override
    public void reset() {
        currentIndex = 0;
    }
    
    @Override
    public int size() {
        return (samples.size() + batchSize - 1) / batchSize;
    }
}