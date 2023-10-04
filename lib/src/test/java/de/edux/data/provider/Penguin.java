package de.edux.data.provider;

/**
 * SeaBorn penguin dto
 */
public record Penguin(String species, String island, double billLengthMm, double billDepthMm, int flipperLengthMm, int bodyMassG, String sex){
    public double[] getFeatures() {
        return new double[]{billLengthMm, billDepthMm, flipperLengthMm, bodyMassG};
    }
}
