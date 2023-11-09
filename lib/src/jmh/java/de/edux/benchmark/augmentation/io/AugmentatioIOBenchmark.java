package de.edux.benchmark.augmentation.io;

import de.edux.augmentation.io.AugmentationImageReader;
import de.edux.augmentation.io.IAugmentationImageReader;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;

@State(Scope.Benchmark)
@OutputTimeUnit(TimeUnit.SECONDS)
@Timeout(time = 25, timeUnit = TimeUnit.SECONDS)
@Warmup(iterations = 1)
public class AugmentatioIOBenchmark {

  IAugmentationImageReader reader;

  public AugmentatioIOBenchmark() {}

  public static void main(String[] args) throws Exception {

    org.openjdk.jmh.Main.main(args);
  }

  @Benchmark
  @Fork(value = 1, warmups = 1)
  @BenchmarkMode({Mode.All})
  public void readBatchOfImages() throws Exception {
    Path benchmarkDataDir =
        Paths.get(
            ".."
                + File.separator
                + "benchmark-data"
                + File.separator
                + "augmentation-benchmark-images");
    reader = new AugmentationImageReader();

    var imageStream = reader.readBatchOfImages(benchmarkDataDir.toString(), 100, 100);
  }
}
