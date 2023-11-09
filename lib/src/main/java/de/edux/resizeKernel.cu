extern "C"
__global__ void resizeKernel(unsigned char* input, unsigned char* output, int originalWidth, int originalHeight, int newWidth, int newHeight)
{
    // Berechne die globale Position des Threads im Output-Array
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < newWidth && y < newHeight)
    {
        // Finde die entsprechende Position im Input-Array
        int oldX = x * originalWidth / newWidth;
        int oldY = y * originalHeight / newHeight;

        // Berechne die Startpunkte für das Input- und Output-Pixel
        int inputIndex = (oldY * originalWidth + oldX) * 3;
        int outputIndex = (y * newWidth + x) * 3;

        // Kopiere die Pixelwerte (für ein RGB-Bild)
        output[outputIndex] = input[inputIndex];     // R
        output[outputIndex + 1] = input[inputIndex + 1]; // G
        output[outputIndex + 2] = input[inputIndex + 2]; // B
    }
}
