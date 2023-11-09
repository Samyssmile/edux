extern "C"
__global__ void scaleImageKernel(unsigned char* input, unsigned char* output, int inputWidth, int inputHeight, int outputWidth, int outputHeight) {
    // Berechne unsere Position im Output-Bild
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= outputWidth || y >= outputHeight) return;

    // Finde die entsprechenden Positionen im Input-Bild
    int nearestX = (int)(x * (inputWidth / (float)outputWidth));
    int nearestY = (int)(y * (inputHeight / (float)outputHeight));

    // Sorge dafür, dass wir nicht über die Grenzen des Input-Bildes hinausgehen
    nearestX = min(nearestX, inputWidth - 1);
    nearestY = min(nearestY, inputHeight - 1);

    // Berechne den Index für die Input- und Output-Pixel
    int inputIndex = (nearestY * inputWidth + nearestX) * 3; // 3 für RGB
    int outputIndex = (y * outputWidth + x) * 3;

    // Kopiere den Pixel vom Input zum Output
    output[outputIndex] = input[inputIndex];     // R
    output[outputIndex + 1] = input[inputIndex + 1]; // G
    output[outputIndex + 2] = input[inputIndex + 2]; // B
}
