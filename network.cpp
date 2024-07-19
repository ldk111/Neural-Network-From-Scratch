#include "network.hpp"

void forward_layer(Layer & curr, Layer & prev) {

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


void NeuralNetwork::train(NeuralNetwork & nn, const std::vector<Matrix> & input_matrix, const std::vector<Matrix> & output_matrix, int n_epoch) {

    for (int epoch = 0; epoch < n_epoch; epoch++) {
        printf("Beginning Epoch: %d\n", epoch + 1);
        for (int i = 0; i < input_matrix.size(); i++) {
            
            Matrix input = input_matrix[i];
            Matrix expected = output_matrix[i];
            
            Matrix& outputs = nn.forward(input);
            
            nn.backpropagate(expected);
            
            //printf("Error: %.4f\n", error(outputs, expected));
        };
        printf("Finished Epoch: %d\n", epoch + 1);
    };
}

double NeuralNetwork::test(NeuralNetwork & nn, const std::vector<Matrix> & input_matrix, const std::vector<Matrix> & output_matrix) {
    
    float count = 0;
    
    for (int i = 0; i < input_matrix.size(); i++) {
    
    Matrix input = input_matrix[i];
    Matrix expected = output_matrix[i];
    
    Matrix& outputs = nn.forward(input);
    double max = 0;
    int val;
    int true_val;
    for (int j = 0; j < 10; j++) {
        if (outputs.at(0, j) > max) {
            max = outputs.at(0, j);
            val = j;
            };
        if (expected.at(0, j) == 1.0) {
            true_val = j;
        };
        };
    //printf("Max: %.4f\n", max);
    //std::cout << val << std::endl;
    //std::cout << true_val<< std::endl;
    if (val == true_val) {
        count += 1;
    };
    };
    return count/(float)(input_matrix.size()+1);
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& config, const std::vector<std::string>& output_labels) {

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
        ::forward_layer(curr, prev);
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
