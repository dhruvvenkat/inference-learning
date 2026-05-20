#include <vector>

using namespace std;

#ifndef MATRIX_HEADER
#define MATRIX_HEADER

class Matrix {
    private:
        vector<vector<int>> values;                         
    
    public:
        Matrix(vector<vector<int>> values);
        ~Matrix();
};


#endif