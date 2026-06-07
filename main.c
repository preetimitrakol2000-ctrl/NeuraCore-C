#include <stdio.h>
#include "neuron.h"

int main() {
    printf("🧠 Initializing NeuraCore-C Inference Layer Engine...\n\n");

    double sample_input[3] = {1.0, 0.5, -0.2};
    double hidden_output[2];
    double final_output[1];

    // Design: 3 Inputs -> 2 Hidden Neurons -> 1 Output Neuron
    Layer hidden_layer = create_layer(3, 2);
    Layer output_layer = create_layer(2, 1);

    forward_layer(&hidden_layer, sample_input, hidden_output);
    forward_layer(&output_layer, hidden_output, final_output);

    printf("📥 Input Tensor: [1.0, 0.5, -0.2]\n");
    printf("🚀 Network Inference Probability Output: %.4f\n", final_output[0]);

    free_layer(&hidden_layer);
    free_layer(&output_layer);
    return 0;
}
