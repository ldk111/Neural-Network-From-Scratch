#pragma once

#include "matrix.hpp"

#include <vector>
#include <filesystem>
#include <fstream>

class Dataset {

    int count;
    Matrix input_matrix;
    Matrix output_matrix;
    Matrix get_input(int index);
    Matrix get_output(int index);

public:

    Dataset(int count, Matrix input_matrix, Matrix output_matrix) {
        count = count;
        input_matrix = input_matrix;
        output_matrix = output_matrix;
    }

};

struct Layer {
    Matrix outputs;
    Matrix biases;
    Matrix weights;

    Layer(int neuron_count = 0);
    Layer next_layer(int neuron_count);
};

void forward(Layer & curr, Layer & prev) {

    curr.outputs = ((prev.outputs * prev.weights) += curr.biases).sigmoid();
};

Layer::Layer(int neuron_count) {

    outputs.init(1, neuron_count);
    biases.init(1, neuron_count);
};

Layer Layer::next_layer(int neuron_count) {

    Layer next;
    next.outputs.init(1, neuron_count);
    next.biases.init(1, neuron_count);
    weights.init(this -> outputs.cols(), next.outputs.cols());
    return next;
};

class NeuralNetwork {

    public:
        Matrix & forward(const Matrix & input);
        void backpropagate(const Matrix & expected); 
        NeuralNetwork(const std::vector<int> & config, const std::vector<std::string> & output_labels);
    private:
        float learn_rate = 0.01;
        std::vector<Layer> layers;
        std::vector<std::string> output_labels;
};

NeuralNetwork::NeuralNetwork(const std::vector<int>& config, const std::vector<std::string>& output_labels) {

  assert(config.size() >= 1);
  assert(output_labels.size() == config.at(config.size() - 1));

  for (size_t i = 0; i < config.size(); i++) {
    int neurons_count = config[i];
    if (i == 0) {
      layers.push_back(Layer(neurons_count));
    } else {
      Layer& prev = layers.at(layers.size() - 1);
      layers.push_back(prev.next_layer(neurons_count));
    }
  }

  for (Layer& layer : layers) {
    layer.weights.randomize(-.5, .5);
  }
}

Matrix & NeuralNetwork::forward(const Matrix & input) {

    layers[0].outputs = input;

    for (int i = 1; i < layers.size(); i++) {
        Layer & curr = layers[i];
        Layer & prev = layers[i - 1];
        ::forward(curr, prev);
    };

    return layers[layers.size() - 1].outputs;
};

float error(Matrix & out, Matrix & exp) {
    return (out - exp).square().sum() / out.cols();
};

void NeuralNetwork::backpropagate(const Matrix& expected) {

  Matrix& output = layers[layers.size() - 1].outputs;
  
  Matrix delta = output - expected; // The difference in y.
    
  for (size_t i = layers.size() - 1; i > 0; i--) {

    Layer& curr = layers[i];

    Layer& prev = layers[i - 1];

    // biases += σ'(z)
    curr.biases += delta * (-learn_rate);

    // weights += prev.Y.T * σ'(z)
    prev.weights += (prev.outputs.transpose() * delta) * (-learn_rate);

    // σ'(z) = σ(z) * (1 - σ(z))
    Matrix one = Matrix(prev.outputs.rows(), prev.outputs.cols(), 1);
    Matrix sigmoid_derivative = prev.outputs.multiply(one - prev.outputs);

    // delta = (delta * prev.W.T) x σ'(z);
    delta = (delta * prev.weights.transpose()).multiply_inplace(sigmoid_derivative);
  }
}
