package de.edux.augmentation;

import de.edux.augmentation.composite.CompositeAugmentationSequence;
import java.util.ArrayList;
import java.util.List;

/** Builder class for creating an augmentation sequence. */
public class AugmentationBuilder {
  private List<AbstractAugmentation> augmentations;

  public AugmentationBuilder() {
    this.augmentations = new ArrayList<>();
  }

  /**
   * Adds an augmentation to the sequence.
   *
   * @param augmentation The augmentation operation to add.
   * @return The builder instance for chaining.
   */
  public AugmentationBuilder addAugmentation(AbstractAugmentation augmentation) {
    this.augmentations.add(augmentation);
    return this;
  }

  /**
   * Builds the final Augmentation sequence from the added operations.
   *
   * @return The composite Augmentation object containing all added operations.
   */
  public AugmentationSequence build() {
    return new CompositeAugmentationSequence(augmentations);
  }
}
