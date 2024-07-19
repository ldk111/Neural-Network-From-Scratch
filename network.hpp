#pragma once

#include "matrix.hpp"

#include <vector>
#include <filesystem>
#include <fstream>

struct Layer {
    Matrix outputs;
    Matrix biases;
    Matrix weights;

    Layer(int neuron_count = 0);
    Layer next_layer(int neuron_count);
};

class NeuralNetwork {

    public:
        Matrix & forward(const Matrix & input);
        void backpropagate(const Matrix & expected); 

        void train(NeuralNetwork & nn, const std::vector<Matrix> & input_matrix, const std::vector<Matrix> & output_matrix, int n_epoch);
        double test(NeuralNetwork & nn, const std::vector<Matrix> & input_matrix, const std::vector<Matrix> & output_matrix);

        NeuralNetwork(const std::vector<int> & config, const std::vector<std::string> & output_labels);

    private:
        float learn_rate = 0.01;
        std::vector<Layer> layers;
        std::vector<std::string> output_labels;
};

