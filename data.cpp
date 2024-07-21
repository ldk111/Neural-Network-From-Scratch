#include "matrix.hpp"
#include "network.hpp"

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>


const int height = 28;
const int width = 28;
int d[width + 1][height + 1];
double expected [11];
double *out1;

std::ifstream image;
std::ifstream label;

void input(std::vector<std::vector<double>> &data, std::vector<double> &expected) {

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
	
    //Displaying image
	//std::cout << "Image:" << std::endl;
	//for (int j = 1; j <= height; ++j) {
	//	for (int i = 1; i <= width; ++i) {
	//		std::cout << d[i][j];
	//	}
	//	std::cout << std::endl;
	//}
    
    //Reading image
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            out1[pos] = d[i][j];
        }
	}
    std::vector<double> row(out1, out1+784);
    
    data.push_back(row);
    
	// Reading label
    label.read(&number, sizeof(char));
    //for (int i = 1; i <= 10; ++i) {
	//	expected[i] = 0.0;
	//}
    //expected[number + 1] = 1.0;

    //std::cout << "Label: " << (int)(number) << std::endl;
    
    expected.push_back((double)(int)(number));
    
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


