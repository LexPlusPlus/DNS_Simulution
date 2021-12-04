#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <omp.h>

void createMatrixEntries(Eigen::VectorXd &b, Eigen::SparseMatrix<double> &A, std::vector<Eigen::Triplet<double>> &tripletList);

int main(){
    
    int n = 10;
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
    double α = -1;
    double β =  1;
    double γ =  1;
    int i = 0;
    
    for(int row = 0; row < A.rows(); row++){

        for(int col = 0; col < A.cols(); col++){
            
            if(row -  1 == col){ // lower diagonal
                tripletList[i] = Eigen::Triplet<double>(row,col,α);
                i++;
            }
            if(row == col){ // main diagonal
                tripletList[i] = Eigen::Triplet<double>(row,col,β);
                i++;
            }
            if(row + 1 == col){ // upper diagonal
                tripletList[i] = Eigen::Triplet<double>(row,col,γ);
                i++;
            }
        }
    }
    b(0) = -α;
    b(A.cols()) = 3; 
    A.setFromTriplets(tripletList.begin(), tripletList.end());
}

