package de.edux.augmentation.benchmarks;

import de.edux.augmentation.io.AugmentationImageReader;
import de.edux.augmentation.io.IAugmentationImageReader;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

public class AugmentatioIOBenchmark {

  IAugmentationImageReader reader;

  public AugmentatioIOBenchmark() {
    reader = new AugmentationImageReader();
  }

  public static void main(String[] args) throws RunnerException {
    Options opt =
        new OptionsBuilder().include(AugmentatioIOBenchmark.class.getSimpleName()).forks(1).build();

    new Runner(opt).run();
  }

  @Benchmark
  @BenchmarkMode(Mode.AverageTime)
  public void readBatchOfImages() throws Exception {
    reader.readBatchOfImages(
        java.nio.file.Paths.get(
            "E:\\projects\\mandelbrot-gan\\MandelbrotGan\\dataset\\class\\tricorn"),
        1000);
  }
}
