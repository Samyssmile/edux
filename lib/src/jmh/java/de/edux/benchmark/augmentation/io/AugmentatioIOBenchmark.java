package de.edux.benchmark.augmentation.io;

import de.edux.augmentation.core.AugmentationBuilder;
import de.edux.augmentation.core.AugmentationSequence;
import de.edux.augmentation.effects.*;
import de.edux.augmentation.effects.geomentry.Perspective;
import de.edux.augmentation.io.AugmentationImageReader;
import de.edux.augmentation.io.IAugmentationImageReader;
import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
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
@Measurement(iterations = 1)
public class AugmentatioIOBenchmark {

  IAugmentationImageReader reader;

  public AugmentatioIOBenchmark() {}

  public static void main(String[] args) throws Exception {
    runBenchmark(ResultFormatType.JSON, "benchmark-augmentation-results.json");
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
  @Measurement(iterations = 1)
  public void readBatchOfImages() throws Exception {
    Path benchmarkDataDir =
        Paths.get(
            ".."
                + File.separator
                + "benchmark-data"
                + File.separator
                + "augmentation-benchmark-images");
    reader = new AugmentationImageReader();

    var imageStream = reader.readImagePathsAsStream(benchmarkDataDir.toString());
  }

  @Benchmark
  @Fork(value = 1, warmups = 1)
  @BenchmarkMode({Mode.All})
  public void imageAugmentation() throws Exception {
    Path benchmarkDataDir =
        Paths.get(
            ".."
                + File.separator
                + "benchmark-data"
                + File.separator
                + "augmentation-benchmark-images");

    Path benchmarkOutputDir =
        Paths.get(
            ".."
                + File.separator
                + "benchmark-data"
                + File.separator
                + "augmentation-benchmark-output");

    AugmentationSequence augmentationSequence =
        new AugmentationBuilder()
            .addAugmentation(new ResizeAugmentation(256, 256, ResizeQuality.QUALITY))
            .addAugmentation(new ContrastAugmentation(0.6f))
            .addAugmentation(new BlurAugmentation(5))
            .addAugmentation(new NoiseInjectionAugmentation(40))
            .addAugmentation(new MonochromeAugmentation())
            .addAugmentation(new BrightnessAugmentation(0.5))
            .addAugmentation(new FlippingAugmentation())
            .addAugmentation(new ColorEqualizationAugmentation())
            .addAugmentation(new CroppingAugmentation(0.8f))
            .addAugmentation(new ElasticTransformationAugmentation(5, 0.5))
            .addAugmentation(new PerspectiveTransformationsAugmentation(Perspective.RIGHT_TILT))
            .addAugmentation(new RandomDeleteAugmentation(5, 10, 10))
            .build()
            .run(benchmarkDataDir.toString(), 4, benchmarkOutputDir.toString());
  }
}
