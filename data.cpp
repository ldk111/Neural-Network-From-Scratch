#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "matrix.hpp"
#include "network.hpp"

const int height = 28;
const int width = 28;
int d[width + 1][height + 1];
double expected [11];
double *out1;

std::ifstream image;
std::ifstream label;
std::ofstream report;

// Training image file name
const std::string training_image_fn = "dataset/train-images.idx3-ubyte";

// Training label file name
const std::string training_label_fn = "dataset/train-labels.idx1-ubyte";

// Testing image file name
const std::string testing_image_fn = "dataset/t10k-images.idx3-ubyte";

// Testing label file name
const std::string testing_label_fn = "dataset/t10k-labels.idx1-ubyte";


void input(std::vector<std::vector<double>> &train_data, std::vector<double> &train_expected) {

    out1 = new double[784 + 1];

	//std::cout << "Reading image" << std::endl;
    char number;
    //std::cout << "Number " << (int)(number) <<std::endl;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[i][j] = 0; 
			} else {
				d[i][j] = 1;
			}
        }
	}
	
	//std::cout << "Image:" << std::endl;
	//for (int j = 1; j <= height; ++j) {
	//	for (int i = 1; i <= width; ++i) {
	//		std::cout << d[i][j];
	//	}
	//	std::cout << std::endl;
	//}
    
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}
    std::vector<double> row(out1, out1+784);
    
    train_data.push_back(row);
    
	// Reading label
    label.read(&number, sizeof(char));
    //for (int i = 1; i <= 10; ++i) {
	//	expected[i] = 0.0;
	//}
    //expected[number + 1] = 1.0;

    //std::cout << "Label: " << (int)(number) << std::endl;
    
    train_expected.push_back((double)(int)(number));
    
}

void train(NeuralNetwork & nn, const std::vector<Matrix> & input_matrix, const std::vector<Matrix> & output_matrix, int n_epoch) {

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

double test(NeuralNetwork & nn, const std::vector<Matrix> & input_matrix, const std::vector<Matrix> & output_matrix) {
    
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

void read_data(std::string image_path, std::string label_path, int n_images, std::vector<Matrix> &input_matrix, std::vector<Matrix> &output_matrix) {

    image.open(image_path.c_str(), std::ios::in | std::ios::binary); // Binary image file
    label.open(label_path.c_str(), std::ios::in | std::ios::binary ); // Binary label file

    std::vector<std::vector<double>> input_vector;
    std::vector<double> input_expected;

    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
    
    for (int sample = 1; sample < n_images; ++sample) {
        input(input_vector, input_expected);
    };   
    
    image.close();
    label.close();
    
    for (int i = 0; i < n_images - 1; ++i) {
        Matrix input_temp;
        Matrix output_temp;
        
        input_temp.init(1, 784, 0.0);
        output_temp.init(1, 10, 0.0);
        
        output_temp.set(0, int(input_expected[i]), 1);
        for (int j = 0; j < 784; ++j){
            input_temp.set(0, j, input_vector[i][j]);
        };
        input_matrix.push_back(input_temp);
        output_matrix.push_back(output_temp);
    };

};

int main(int argc, char *argv[]){

    int n_images = 1000;

    std::vector<std::vector<double>> input_vector;
    std::vector<double> input_expected;

    std::vector<Matrix> input_matrix_train;
    std::vector<Matrix> output_matrix_train;
    
    read_data(training_image_fn, training_label_fn, 60000, input_matrix_train, output_matrix_train);

    // Create layers.
    Layer input_layer(784);
    Layer hidden_1 = input_layer.next_layer(16);
    Layer hidden_2 = hidden_1.next_layer(10);
    Layer output   = hidden_2.next_layer(10);
    
    // Construct neural network.
    NeuralNetwork nn({784, 20, 10, 10}, {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"});
    
    // Train the model.
    train(nn, input_matrix_train, output_matrix_train, 16);

    std::vector<Matrix> input_matrix_test;
    std::vector<Matrix> output_matrix_test;

    read_data(testing_image_fn, testing_label_fn, 10000, input_matrix_test, output_matrix_test);

    printf("Percentage Accuracy: %.4f\n", test(nn, input_matrix_test, output_matrix_test));

    return 0;
}

