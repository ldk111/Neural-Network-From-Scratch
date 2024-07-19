#include "matrix.hpp"

#include <stdlib.h>
#include <math.h>
#include <iostream>


Matrix::Matrix(int rows, int cols, double val): _rows(rows), _cols(cols), _data(rows* cols, val){}


Matrix& Matrix::init(int rows, int cols, double val) {
  this->_rows = rows;
  this->_cols = cols;
  this->_data = std::vector<double>(rows * cols, val);
  return *this;
}

double Matrix::at(int row, int col) const {
  return _data[row * _cols + col];
}

void Matrix::set(int row, int col, double value) {
  _data[row * _cols + col] = value;
}

Matrix& Matrix::fill(double val) {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = val;
  }
  return *this;
}

Matrix& Matrix::randomize(double min, double max) {

  for (size_t i = 0; i < _data.size(); i++) {
    double val = (double)((double)rand() / (double)RAND_MAX) * (max - min) + min;
    _data[i] = val;
  }
  return *this;
}


double Matrix::sum() const {
  double total = 0;
  for (size_t i = 0; i < _data.size(); i++) {
    total += _data[i];
  }
  return total;
}

static inline double sigmoid(double x) {
  return 1.f / (1.f + expf(-x));
}


Matrix& Matrix::sigmoid() {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = ::sigmoid(_data[i]);
  }
  return *this;
}



Matrix& Matrix::square() {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] = _data[i] * _data[i];
  }
  return *this;
}


std::vector<double>& Matrix::data() {
  return _data;
}


const std::vector<double>& Matrix::data() const {
  return _data;
}


int Matrix::rows() const {
  return _rows;
}


int Matrix::cols() const {
  return _cols;
}


void Matrix::print() const {
  printf("[");
  for (int r = 0; r < _rows; r++) {
    for (int c = 0; c < _cols; c++) {
      if (c != 0) printf(", ");
      // negative number has extra '-' character at the start.
      double val = at(r, c);
      if (val >= 0) printf(" %.6f", val);
      else printf("%.6f", val);
    }
    printf("  ");
    printf("\n");
    printf("\n");
  }
  printf("]\n");
}


Matrix& Matrix::operator+=(const Matrix& other) {
  bool cond = (_rows == other._rows && _cols == other._cols);
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] += other._data[i];
  }
  return *this;
}


Matrix& Matrix::operator*=(double value) {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] *= value;
  }
  return *this;
}


Matrix& Matrix::multiply_inplace(const Matrix& other) {
  for (size_t i = 0; i < _data.size(); i++) {
    _data[i] *= other._data[i];
  }
  return *this;
}


Matrix Matrix::operator-(const Matrix& other) const {
  Matrix m(this->_rows, this->_cols);
  for (size_t i = 0; i < _data.size(); i++) {
    m._data[i] = this->_data[i] - other._data[i];
  }
  return m;
}


Matrix Matrix::operator*(const Matrix& other) const {

  // (r1 x c1) * (r2 x c2) =>
  //   assert(c1 == r2), result = (r1 x c2)
  //assert(this->_cols == other._rows);

  Matrix m(this->_rows, other._cols);

  int n = _cols; // Width or a row.
  for (int r = 0; r < m._rows; r++) {
    for (int c = 0; c < m._cols; c++) {

      double val = 0;
      for (int i = 0; i < n; i++) {
        val += this->at(r, i) * other.at(i, c);
      }
      m.set(r, c, val);
    }
  }

  return m;
}


Matrix Matrix::operator*(double value) const {
  Matrix m(_rows, _cols);
  std::vector<double>& m_data = m.data();
  for (size_t i = 0; i < _data.size(); i++) {
    m_data[i] = _data[i] * value;
  }
  return m;
}


Matrix Matrix::transpose() const {
  Matrix m(_cols, _rows);
  for (int r = 0; r < _rows; r++) {
    for (int c = 0; c < _cols; c++) {
      m.set(c, r, at(r, c));
    }
  }
  return m;
}


Matrix Matrix::multiply(const Matrix& other) const {
  Matrix m(_rows, _cols);
  for (size_t i = 0; i < _data.size(); i++) {
    m._data[i] = _data[i] * other._data[i];
  }
  return m;
}
