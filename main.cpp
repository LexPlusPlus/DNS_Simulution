#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse> 

#include <omp.h>

void createMatrixEntries(Eigen::VectorXd &b, Eigen::SparseMatrix<double> &A, std::vector<Eigen::Triplet<double>> &tripletList);

int main(){
    
    int n = 5;
    Eigen::VectorXd b(n), x(n);
    Eigen::SparseMatrix<double> A(n,n);
    std::vector<Eigen::Triplet<double>> tripletList(3*n - 2);

    tripletList.reserve(3*n - 2);
    createMatrixEntries(b, A, tripletList);
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    x = solver.solve(b);
    
    std::cout << A << std::endl;
    std::cout << b << std::endl;
    std::cout << "\nresult:\n" << x << std::endl;
}

void createMatrixEntries(Eigen::VectorXd &b, Eigen::SparseMatrix<double> &A, std::vector<Eigen::Triplet<double>> &tripletList){
    // Γ = Gamma, Φ = Phi, ρ = rho.
    // D_i = (Γ/δx)_i, F_i = (ρu)_i
    double alpha = -1;
    double beta =  1;
    double gamma =  1;
    int i = 0;
    
    for(int row = 0; row < A.rows(); row++){

        for(int col = 0; col < A.cols(); col++){
                        
            if(row -  1 == col){ // lower diagonal
                tripletList[i] = Eigen::Triplet<double>(row,col,alpha);
                i++;
            }
            if(row == col){ // main diagonal
                tripletList[i] = Eigen::Triplet<double>(row,col,beta);
                i++;
            }
            if(row + 1 == col){ // upper diagonal
                tripletList[i] = Eigen::Triplet<double>(row,col,gamma);
                i++;
            }
        }
    }
    b(0) = -alpha;
    // b(A.cols() - 1) = 3; 
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

