#include <stdlib.h>
#include <math.h>
#include "neuron.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

Layer create_layer(int input_dim, int output_dim) {
    Layer layer;
    layer.input_dim = input_dim;
    layer.output_dim = output_dim;
    
    layer.weights = (double*)malloc(input_dim * output_dim * sizeof(double));
    layer.biases = (double*)malloc(output_dim * sizeof(double));
    
    // Initialize with static weights for predictability (simulate pre-trained edge model)
    for (int i = 0; i < input_dim * output_dim; i++) layer.weights[i] = 0.5;
    for (int i = 0; i < output_dim; i++) layer.biases[i] = -0.1;
    
    return layer;
}

void forward_layer(Layer* layer, double* inputs, double* outputs) {
    for (int j = 0; j < layer->output_dim; j++) {
        double activation = layer->biases[j];
        for (int i = 0; i < layer->input_dim; i++) {
            activation += inputs[i] * layer->weights[i * layer->output_dim + j];
        }
        outputs[j] = sigmoid(activation);
    }
}

void free_layer(Layer* layer) {
    free(layer->weights);
    free(layer->biases);
}
