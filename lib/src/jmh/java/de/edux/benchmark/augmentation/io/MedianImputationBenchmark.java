package de.edux.benchmark.augmentation.io;

import de.edux.functions.imputation.MedianImputation;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.profile.GCProfiler;
import org.openjdk.jmh.profile.MemPoolProfiler;
import org.openjdk.jmh.profile.StackProfiler;
import org.openjdk.jmh.results.format.ResultFormatType;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

@State(Scope.Benchmark)
@OutputTimeUnit(TimeUnit.SECONDS)
@Timeout(time = 25, timeUnit = TimeUnit.SECONDS)
@Warmup(iterations = 1)
@Measurement(iterations = 2)
public class MedianImputationBenchmark {
  public static void main(String[] args) throws Exception {
    runBenchmark(ResultFormatType.JSON, "benchmark-median-imputation-results.json");
  }

  private static void runBenchmark(ResultFormatType formatType, String fileName) throws Exception {
    Options opt =
        new OptionsBuilder()
            .include(AugmentatioIOBenchmark.class.getSimpleName())
            .resultFormat(formatType)
            .result("benchmark-data/results/" + fileName)
            .addProfiler(GCProfiler.class)
            .addProfiler(MemPoolProfiler.class)
            .addProfiler(StackProfiler.class)
            .build();

    new Runner(opt).run();
  }

  @Benchmark
  @Fork(value = 1, warmups = 1)
  @BenchmarkMode({Mode.All})
  public void readBatchOfImages() throws Exception {

    String[] largeDataset = new String[1000000];
    Random random = new Random();
    for (int i = 0; i < largeDataset.length; i++) {
      if (random.nextDouble() < 0.05) { // 5% empty values
        largeDataset[i] = "";
      } else {
        largeDataset[i] = String.valueOf(random.nextDouble() * 1000000);
      }
    }

    double[] numericValues =
        Arrays.stream(largeDataset)
            .filter(s -> !s.isBlank())
            .mapToDouble(Double::parseDouble)
            .sorted()
            .toArray();
    double expectedMedian =
        numericValues.length % 2 == 0
            ? (numericValues[numericValues.length / 2]
                    + numericValues[numericValues.length / 2 - 1])
                / 2.0
            : numericValues[numericValues.length / 2];

    MedianImputation medianImputation = new MedianImputation();

    long startTime = System.nanoTime();
    double calculatedMedian = medianImputation.calculateMedian(largeDataset);
    long endTime = System.nanoTime();

    System.out.println("Process time in seconds: " + (endTime - startTime) / 1e9);
  }
}
