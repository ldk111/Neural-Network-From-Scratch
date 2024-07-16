#include <vector>
#include <stdlib.h>
#include <time.h>

#include "matrix.hpp"
#include "network.hpp"

int main() {

  // Load training dataset.
  MNIST dataset_train(
    "dataset/train-labels.idx1-ubyte",
    "dataset/train-images.idx3-ubyte");

  // Create layers.
  Layer input(784);
  Layer hidden_1 = input.next_layer(16);
  Layer hidden_2 = hidden_1.next_layer(10);
  Layer output   = hidden_2.next_layer(10);

  // Construct neural network.
  NeuralNetwork nn({784, 20, 10, 10}, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});

  // Train the model.
  train(nn, dataset_train, /*epoch=*/3);

  return 0;
}