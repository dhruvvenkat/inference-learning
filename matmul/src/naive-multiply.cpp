#include <iostream>
#include "matrix.hpp"
#include <vector>

using namespace std;

Matrix::Matrix(vector<vector<int>> input) {
    values = input;
}

Matrix::~Matrix() {
}

vector<vector<int>> Matrix::getVals() {
    return values;
}

void printMatrix(Matrix& matrixIn) {
    vector<vector<int>> vals = matrixIn.getVals();
    for (long unsigned int i = 0; i < vals.size(); i++) {
        for (long unsigned int j = 0; j < vals[i].size(); j++) {
            cout << vals[i][j] << " ";
        }
        cout << endl;
    }

}

Matrix naive(Matrix& matrix1, Matrix& matrix2) {
    vector<vector<int>> vals1 = matrix1.getVals();
    vector<vector<int>> vals2 = matrix2.getVals();
    vector<int> currRow;
    vector<vector<int>> output;

    for (int k = 0; k < vals1.size(); k++) {
        for (int i = 0; i < vals2[0].size(); i++) {
            int currCell = 0;
            for (int j = 0; j < vals1[0].size(); j++) {
                currCell += vals1[k][j] * vals2[j][i];
            }
            currRow.push_back(currCell);
        }
        output.push_back(currRow);
        currRow.clear();
    }

    Matrix outputMatrix = Matrix(output);
    return outputMatrix;
}

int main() {
    vector<vector<int>> v1 = {{1, 2, 3}, {4, 5, 6}};
    Matrix matrix1 = Matrix(v1);

    vector<vector<int>> v2 = {{7, 8}, {9, 10}, {11, 12}};
    Matrix matrix2 = Matrix(v2);
    
    printMatrix(matrix1);
    cout << "////////" << endl;
    printMatrix(matrix2);
    cout << "////////" << endl;

    Matrix output = naive(matrix1, matrix2);

    printMatrix(output);

}

