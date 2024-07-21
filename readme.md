##Multi-Layer Perceptron Neural Network with C++ Standard Libraries

This is a project I have developed to help me better understand both C++ and neural networks. 

I have created a multi-layer perceptron network that trains on the MNIST dataset to identify numbers from black and white images. This network class can be extended to any task a simple multi-layer perceptron can be used for by modifying the input and output vectors in main.cpp. The data.cpp file is made specifically for dealing with the MNIST dataset so is not needed if other input vectors are going to be used, the network itself is contained entirely within network.cpp and matrix.cpp, and accessed via main.cpp.

To compile and execute I used gcc with the following instructions.

    g++ -c matrix.cpp
    g++ -c network.cpp
    g++ -c data.cpp
    g++ -c main.cpp
    g++ main.o data.o network.o matrix.o
    ./a.out