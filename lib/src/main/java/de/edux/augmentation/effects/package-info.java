/**
 * This package contains classes that perform various image augmentation effects. Image augmentation
 * is a technique used to increase the diversity of your training set by applying random
 * transformations such as cropping, padding, and horizontal flipping, among others. These
 * transformations are a form of regularization that helps prevent overfitting and allows your model
 * to generalize better.
 *
 * <p>The 'effects' subpackage specifically includes classes that implement different kinds of
 * transformations, like resizing, adjusting perspective, and applying elastic distortions to
 * images. Each class extends the 'AbstractAugmentation' class and overrides its 'apply' method to
 * provide the specific functionality.
 */
package de.edux.augmentation.effects;
