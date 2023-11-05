package de.edux.data.provider.experimental;

public class DataElement {
    private String label;
    private String imagePath;

    public DataElement(String label, String imagePath) {
        this.label = label;
        this.imagePath = imagePath;
    }

    public String getLabel() {
        return label;
    }

    public void setLabel(String label) {
        this.label = label;
    }

    public String getImagePath() {
        return imagePath;
    }

    public void setImagePath(String imagePath) {
        this.imagePath = imagePath;
    }
}
