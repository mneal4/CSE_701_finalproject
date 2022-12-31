#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
using namespace std;

/**
 * @brief Class for matrix operations.
 * Please note that much of the matrix class came from lecture notes.
 * Up until line 168 comes from lecture notes, after that are functions I added.
 */
class matrix
{
public:
    /**
     * @brief Construct a new zero matrix, this came from lecture notes
     * This came from lecture notes.
     * @param uint64_t of row number
     * @param uint64_t of col number
     */

    matrix(const uint64_t &, const uint64_t &);

    /**
     * @brief Construct a new matrix object
     * This came from lecture notes.
     * @param uint64_t of row number
     * @param uint64_t of col number
     * @param vector<double> of matrix elements
     */

    matrix(const uint64_t &, const uint64_t &, const vector<double> &);

    /**
     * @brief Member function to obtain (but not modify) the number of rows in the matrix.
     * This came from lecture notes.
     * @return Number of rows.
     */
    uint64_t get_rows() const;

    /**
     * @brief Member function to obtain (but not modify) the number of columns in the matrix.
     * This came from lecture notes.
     * @return Number of columns.
     */
    uint64_t get_cols() const;

    /**
     * @brief Overloaded operator () to access matrix elements WITHOUT range checking,
     * and allows modification of the element.
     * This came from lecture notes.
     * @return Returns a matrix element.
     */
    double &operator()(const uint64_t &, const uint64_t &);

    /**
     * @brief Overloaded operator () to access matrix elements WITHOUT range checking,
     * and does not allow modification of the element.
     * This came from lecture notes.
     * @return Returns a matrix element.
     */
    const double &operator()(const uint64_t &, const uint64_t &) const;

    /**
     * @brief Exception to be thrown if the number of rows or columns given to the constructor is zero.
     * This came from lecture notes.
     */
    inline static invalid_argument zero_size = invalid_argument("Matrix cannot have zero rows or columns!");

    /**
     * @brief Exception to be thrown if the vector of elements provided to the constructor is of the wrong size.
     * This came from lecture notes.
     */
    inline static invalid_argument initializer_wrong_size = invalid_argument("Initializer does not have the expected number of elements!");

    /**
     * @brief Exception to be thrown if two matrices of different sizes are added or subtracted.
     * This came from lecture notes.
     */
    inline static invalid_argument incompatible_sizes_add = invalid_argument("Cannot add or subtract two matrices of different dimensions!");

    /**
     * @brief Exception to be thrown if two matrices of incompatible sizes are multiplied.
     * This came from lecture notes.
     */
    inline static invalid_argument incompatible_sizes_multiply = invalid_argument("Two matrices can only be multiplied if the number of columns in the first matrix is equal to the number of rows in the second matrix!");

    // Overloaded Operators for Matrix Arithmetic
    // Please note that these overloaded operators come from the lecture notes.

    /**
     * @brief Friend function to print matrix.
     *
     * @return Prints out matrix.
     */
    friend ostream &operator<<(ostream &, const matrix &);

    /**
     * @brief Overloaded binary operator to add to matrices.
     *
     * @return Returns a resulting matrix from addition.
     */
    matrix operator+(const matrix &);

    /**
     * @brief Overloaded binary operator to perform matrix addition and assignment.
     *
     * @return Returns resulting matrix.
     */
    friend matrix operator+=(matrix &, const matrix &);

    /**
     * @brief Overloaded unary operator to negate a matrix.
     *
     * @return Returns resulting matrix.
     */
    friend matrix operator-(const matrix &);

    /**
     * @brief Overloaded binary operator to subtract to matrices.
     *
     * @return Returns resulting matrix.
     */
    matrix operator-(const matrix &);

    /**
     * @brief Overloaded binary operator to perform matrix subtraction and assignment.
     *
     * @return Returns resulting matrix.
     */
    friend matrix operator-=(matrix &, const matrix &);

    /**
     * @brief Overloaded binary operator to multiply two matrices.
     *
     * @return Returns resulting matrix.
     */
    matrix operator*(const matrix &);

    /**
     * @brief Overloaded binary operator to perform scalar multiplication on the right.
     *
     * @return Returns resulting matrix.
     */
    matrix operator*(const double &);
    /**
     * @brief Overloaded operator to perform scalar division on the right.
     *
     * @return Returns resulting matrix.
     */
    matrix operator/(const double &);

    /**
     * @brief Friend function of overloaded operator to perform scalar multiplication on the right.
     *
     * @return Returns resulting matrix.
     */
    friend matrix operator*(const matrix &, const double &);
    /**
     * @brief Friend function of overloaded operator to perform scalar multiplication on the left.
     *
     * @return Returns resulting matrix.
     */
    friend matrix operator*(const double &s, const matrix &m);

    // This is the end of matrix functions that came from lecture notes. The remaining functions in this class were added by me.
    //  Extra Matrix Functions

    /**
     * @brief Function to extract a row from a matrix.
     *
     * @param row
     * @return Returns row matrix.
     */
    matrix extract_row(const matrix &, uint64_t row);
    /**
     * @brief Function to extract a column from a matrix.
     *
     * @param col
     * @return Returns a column matrix.
     */
    matrix extract_col(const matrix &, uint64_t col);
    /**
     * @brief Function to calculate the determinant of a matrix. This was adapted from my assignment 1 submission.
     *
     * @param rows
     * @param cols
     * @return Returns the determinant of a matrix as a double.
     */
    double determinant(uint64_t rows, uint64_t cols, const matrix &);
    /**
     * @brief Function to calculate the cofactor of a matrix. This is used in calculating the inverse of a matrix.
     * This function was adapted from my project 1 submission.
     * @param rows
     * @param cols
     * @param p
     * @param q
     * @return Returns a cofactor matrix based on p and q.
     */
    matrix cofactor(uint64_t rows, uint64_t cols, const matrix &, uint64_t p, uint64_t q);
    /**
     * @brief Function to calculate the inverse of a matrix, using the adjoint and determinant.
     * This code was adapted from determinant code.
     * @return Returns the inverse of a matrix.
     */
    matrix inverse(const matrix &);
    /**
     * @brief Function to calculate a mean matrix from a matrix.
     * The calculation of the adjoint come from https://www.geeksforgeeks.org/adjoint-inverse-matrix/ with some modifications.
     * @param rows
     * @param cols
     * @return Returns a mean matrix.
     */
    matrix mean(uint64_t rows, uint64_t cols, const matrix &);
    /**
     * @brief Function to find the covariance matrix from a matrix.
     *
     * @param rows
     * @param cols
     * @return Returns a covariance matrix.
     */
    matrix covariance_matrix(uint64_t rows, uint64_t cols, const matrix &);
    /**
     * @brief Function to find the transpose of a matrix.
     *
     * @param rows
     * @param cols
     * @return Returns the transpose of a matrix.
     */
    matrix transpose(uint64_t rows, uint64_t cols, const matrix &);
    /**
     * @brief Function to create a zero matrix.
     *
     * @param rows
     * @param cols
     * @return Returns a zero matrix.
     */
    matrix zero_matrix(uint64_t rows, uint64_t cols);
    // End of functions that were added to this class

private:
    /**
     * @brief The number of rows. This comes from lecture notes.
     *
     */
    uint64_t rows = 0;

    /**
     * @brief The number of columns. This comes from lecture notes.
     *
     */
    uint64_t cols = 0;

    /**
     * @brief A vector storing the elements of a matrix in 1-dimensional form.
     * Ths comes from lecture notes.
     */
    vector<double> elements;
};

/**
 * @brief Class for EM algorithm
 * This is class is derived from the matrix class to allow for matrix objects and arithmetic to be implemented.
 */
class EMalgo : public matrix
{

public:
    /**
     * @brief Function to calculate the density of a multivariate normal.
     *
     * @param data
     * @param mu
     * @param cov
     * @return Returns a matrix containing the density of a multivariate normal.
     */
    matrix dmvnorm(matrix &data, matrix &mu, matrix &cov);
    /**
     * @brief Function to calculate the complete data log-likelihood of a Gaussian mixture model.
     *
     * @param data
     * @param pigs
     * @param mus
     * @param sigmas
     * @param G
     * @return Returns a matrix containing the complete data log-likelihood.
     */
    double compute_log_likelihood(matrix &data, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G);
    /**
     * @brief Function to create a vector of matrices. This allows for mean, vector, and probability matrices to be
     * stored by group number.
     * @param G
     * @param rows
     * @param cols
     * @return Returns a vector of matrices.
     */
    vector<matrix> vec_mat(uint64_t G, uint64_t rows, uint64_t cols);
    /**
     * @brief Function to perform the E step of the EM algorithm.
     *
     * @param data
     * @param pigs
     * @param mus
     * @param sigmas
     * @param G
     * @return Returns a matrix containing soft classifications.
     */
    matrix e_step(matrix &data, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G);
    /**
     * @brief Function to perform the M step of the EM algorithm.
     *
     * @param data
     * @param zigs
     * @param pigs
     * @param mus
     * @param sigmas
     * @param G
     * @return Matrices are updated but nothing is returned.
     */
    void m_step(matrix &data, matrix &zigs, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G);

    /**
     * @brief Function to provide random soft inital classifications.
     *
     * @param rows
     * @return Returns random soft classifications.
     */
    matrix random_soft(uint64_t rows);
    /**
     * @brief Function to implement the EM algorithm.
     *
     * @param data
     * @param zigs
     * @param pigs
     * @param mus
     * @param sigmas
     * @param G
     * @return Returns estimated soft classifications.
     */
    matrix EM(matrix &data, matrix &zigs, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G);
    /**
     * @brief Function to convert soft classifications to hard classifications
     *
     * @param zigs
     * @return Returns hard classifications.
     */
    matrix classify(matrix &zigs);
    /**
     * @brief Function to calculate the adjusted Rand index.
     *
     * @param classified
     * @param true_class
     * @return Returns ARI as a double.
     */
    double ARI(matrix &classified, matrix &true_class);
    /**
     * @brief Function to calculate the binomial coefficients needed to calculate the ARI.
     * This function comes from https://www.geeksforgeeks.org/binomial-coefficient-dp-9/

     * @param n
     * @param k
     * @return Returns the binomial coefficient as a double.
     */
    double binomial_coeff(double n, double k);
};
