#pragma once

#include <vector>

class Matrix {

    public:
        Matrix(int rows = 0, int cols = 0, double val = 0);
        Matrix & init(int rows, int cols, double val = 0);
        Matrix & fill(double val = 0);

        void print() const;

        Matrix & operator+=(const Matrix & other);
        Matrix & operator*=(double value);
        Matrix & multiply_inplace(const Matrix & other);

        Matrix operator-(const Matrix & other) const;
        Matrix operator*(const Matrix & other) const;
        Matrix operator*(double value) const;
        Matrix transpose() const;
        Matrix multiply(const Matrix & other) const;

        double at(int row, int col) const;
        void set(int row, int col, double value);

        double sum() const;
        Matrix& randomize(double min = 0, double max = 1);
        Matrix& sigmoid();
        Matrix& square();

        std::vector<double>& data();
        const std::vector<double>& data() const;

        int rows() const;
        int cols() const;

    private:
        int _rows, _cols;
        std::vector<double> _data;
};




