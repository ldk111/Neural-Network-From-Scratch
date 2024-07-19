#pragma once

#include "matrix.hpp"
#include "network.hpp"

#include <string>
#include <vector>

void input(std::vector<std::vector<double>> &data, std::vector<double> &expected);

void read_data(std::string image_path, std::string label_path, int n_images, std::vector<Matrix> &input_matrix, std::vector<Matrix> &output_matrix);