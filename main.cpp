#include <vector>
#include <stdlib.h>
#include <time.h>

#include "matrix.hpp"
#include "network.hpp"
#include "data.hpp"

// Training image file name
const std::string training_image = "dataset/train-images.idx3-ubyte";

// Training label file name
const std::string training_label = "dataset/train-labels.idx1-ubyte";

// Testing image file name
const std::string testing_image = "dataset/t10k-images.idx3-ubyte";

// Testing label file name
const std::string testing_label = "dataset/t10k-labels.idx1-ubyte";

int main(int argc, char *argv[]){

    int n_images = 1000;

    std::vector<std::vector<double>> input_vector;
    std::vector<double> input_expected;

    std::vector<Matrix> input_matrix_train;
    std::vector<Matrix> output_matrix_train;
    
    read_data(training_image, training_label, 60000, input_matrix_train, output_matrix_train);

    // Create layers.
    Layer input_layer(784);
    Layer hidden_1 = input_layer.next_layer(16);
    Layer hidden_2 = hidden_1.next_layer(10);
    Layer output   = hidden_2.next_layer(10);
    
    // Construct neural network.
    NeuralNetwork nn({784, 20, 10, 10}, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
    
    // Train the model.
    nn.train(nn, input_matrix_train, output_matrix_train, 16);

    std::vector<Matrix> input_matrix_test;
    std::vector<Matrix> output_matrix_test;

    read_data(testing_image, testing_label, 10000, input_matrix_test, output_matrix_test);

    printf("Percentage Accuracy: %.4f\n", nn.test(nn, input_matrix_test, output_matrix_test));

    return 0;
}
