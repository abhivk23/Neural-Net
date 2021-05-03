// Abhiram Kakuturu
// Sandbox for dev in Eigen library
#include <iostream>
#include <eigen3/Eigen/Dense>
using namespace Eigen;
using std::cout;
using std::endl;

int main() {
/*
    // Initialization and Construction //
    // There exist predefined 'fixed' matrices that we can use
    Vector4f v4f; // initialize 4x1 column vector of floats
    v4f << 1.0f, 4.0f, 3.0f, 2.0f; // declare via comma-initialization
    cout << v4f << endl;

    Matrix4f m4d; // initialize 4x4 matrix of doubles
    for(int i=0; i<m4d.rows(); i++){
        for(int j=0; j<m4d.cols(); j++){
            m4d(i,j) = i + j +1 ; // declare by accessing element
        }
    }
    cout << m4d << endl;
    cout << m4d * v4f << endl;

    // Can also construct dynamic matrices whose dimensions are unknown at compile time
    Matrix<float, 3, Dynamic> I;
    I.resize(3,3); // but in order to declare, we need to define dimensions via resize
    I << 1.0f, 0.0f, 0.0f,
         0.0f, 1.0f, 0.0f,
         0.0f, 0.0f, 1.0f;
    cout << I << endl;
    MatrixXf A(4,5); // Immediately define row and col sizes for arbitrary dynamic matrix with all zeros
    MatrixXf C; // Initially a 0x0 initialized matrix via predefined typedef
    C.resize(4,5); // equivalent to A after resizing
    cout << C << endl;

    // Linear Algebra //
    cout << A.transpose() << endl; // matrix transpose
    cout << I.inverse() << endl; // matrix inverse, NaNs if not invertible (det(M)=0)
    
    Vector4f v4f(1.0f, 0.0f, 0.0f, 1.0f);
    cout << "Norm of (" << v4f.transpose() << ") = " << v4f.dot(v4f) << endl;
    cout << "Equivalently: " << v4f.transpose()*v4f << endl;
*/
    Eigen::Matrix<float, 2,2> train_data;
    train_data << 1.0, 0.0,
                  1.0, -5.0;
    //cout << train_data << endl;
    cout << train_data.cwiseAbs() << endl;
}