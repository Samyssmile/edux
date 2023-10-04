package de.edux.data.provider;

import java.util.ArrayList;
import java.util.List;

public class SeabornDataProcessor extends DataUtil<Penguin> {
    @Override
    public void normalize(List<Penguin> penguins) {
        double maxBillLength = penguins.stream().mapToDouble(Penguin::billLengthMm).max().orElse(1);
        double minBillLength = penguins.stream().mapToDouble(Penguin::billLengthMm).min().orElse(0);

        double maxBillDepth = penguins.stream().mapToDouble(Penguin::billDepthMm).max().orElse(1);
        double minBillDepth = penguins.stream().mapToDouble(Penguin::billDepthMm).min().orElse(0);

        double maxFlipperLength = penguins.stream().mapToInt(Penguin::flipperLengthMm).max().orElse(1);
        double minFlipperLength = penguins.stream().mapToInt(Penguin::flipperLengthMm).min().orElse(0);

        double maxBodyMass = penguins.stream().mapToInt(Penguin::bodyMassG).max().orElse(1);
        double minBodyMass = penguins.stream().mapToInt(Penguin::bodyMassG).min().orElse(0);

        List<Penguin> normalizedPenguins = new ArrayList<>();
        for (Penguin p : penguins) {
            double normalizedBillLength = (p.billLengthMm() - minBillLength) / (maxBillLength - minBillLength);
            double normalizedBillDepth = (p.billDepthMm() - minBillDepth) / (maxBillDepth - minBillDepth);
            double normalizedFlipperLength = (p.flipperLengthMm() - minFlipperLength) / (maxFlipperLength - minFlipperLength);
            double normalizedBodyMass = (p.bodyMassG() - minBodyMass) / (maxBodyMass - minBodyMass);

            p = new Penguin(p.species(), p.island(), normalizedBillLength, normalizedBillDepth, (int) normalizedFlipperLength, (int) normalizedBodyMass, p.sex());

            Penguin normalizedPenguin = new Penguin(p.species(), p.island(), normalizedBillLength, normalizedBillDepth, (int) normalizedFlipperLength, (int) normalizedBodyMass, p.sex());
            normalizedPenguins.add(normalizedPenguin);
        }
        penguins.clear();
        penguins.addAll(normalizedPenguins);
    }

    @Override
    public Penguin mapToDataRecord(String[] csvLine) {
        if (csvLine.length != 7) {
            throw new IllegalArgumentException("CSV line format is invalid. Expected 7 fields, got " + csvLine.length + ".");
        }

        for (String value : csvLine) {
            if (value == null || value.trim().isEmpty()) {
                return null;
            }
        }

        String species = csvLine[0];
        if (!(species.equalsIgnoreCase("adelie") || species.equalsIgnoreCase("chinstrap") || species.equalsIgnoreCase("gentoo"))) {
            throw new IllegalArgumentException("Invalid species: " + species);
        }

        String island = csvLine[1];

        double billLengthMm;
        double billDepthMm;
        int flipperLengthMm;
        int bodyMassG;

        try {
            billLengthMm = Double.parseDouble(csvLine[2]);
            if (billLengthMm < 0) {
                throw new IllegalArgumentException("Bill length cannot be negative.");
            }

            billDepthMm = Double.parseDouble(csvLine[3]);
            if (billDepthMm < 0) {
                throw new IllegalArgumentException("Bill depth cannot be negative.");
            }

            flipperLengthMm = Integer.parseInt(csvLine[4]);
            if (flipperLengthMm < 0) {
                throw new IllegalArgumentException("Flipper length cannot be negative.");
            }

            bodyMassG = Integer.parseInt(csvLine[5]);
            if (bodyMassG < 0) {
                throw new IllegalArgumentException("Body mass cannot be negative.");
            }
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException("Invalid number format in CSV line", e);
        }

        String sex = csvLine[6];
        if (!(sex.equalsIgnoreCase("male") || sex.equalsIgnoreCase("female"))) {
            throw new IllegalArgumentException("Invalid sex: " + sex);
        }

        return new Penguin(species, island, billLengthMm, billDepthMm, flipperLengthMm, bodyMassG, sex);
    }

    @Override
    public double[][] getInputs(List<Penguin> dataset) {
        double[][] inputs = new double[dataset.size()][4];

        for (int i = 0; i < dataset.size(); i++) {
            Penguin p = dataset.get(i);
            inputs[i][0] = p.billLengthMm();
            inputs[i][1] = p.billDepthMm();
            inputs[i][2] = p.flipperLengthMm();
            inputs[i][3] = p.bodyMassG();
        }

        return inputs;
    }

    @Override
    public double[][] getTargets(List<Penguin> dataset) {
        double[][] targets = new double[dataset.size()][1];

        for (int i = 0; i < dataset.size(); i++) {
            Penguin p = dataset.get(i);
            targets[i][0] = "Male".equalsIgnoreCase(p.sex()) ? 1.0 : 0.0;
        }

        return targets;
    }
}