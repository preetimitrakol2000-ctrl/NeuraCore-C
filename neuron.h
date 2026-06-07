#ifndef NEURON_H
#define NEURON_H

typedef struct {
    int input_dim;
    int output_dim;
    double* weights;
    double* biases;
} Layer;

Layer create_layer(int input_dim, int output_dim);
void forward_layer(Layer* layer, double* inputs, double* outputs);
double sigmoid(double x);
void free_layer(Layer* layer);

#endif
