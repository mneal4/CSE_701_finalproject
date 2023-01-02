/**
 * @file EMalgo.cpp
 * @author Mackenzie Neal (nealm6@mcmaster.ca)
 * @brief A program to perform the EM algorithm for Gaussian mixture model-based clustering
 * @version 0.1
 * @date 2022-12-21
 *
 * @copyright Copyright (c) 2022
 *
 */
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cfloat>
#include <numbers>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cinttypes>
#include <string>
#include <random>
using namespace std;

// Please note that I make use of the matrix class in this project
// Thus the matrix construction and overloaded operators come from lecture notes.
// You will see a comment signalling where the matrix stuff from lecture ends.

// Classes
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
    matrix random_soft(uint64_t rows, uint64_t G, uint64_t seedval);
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

// End of classes

/**
 * @brief Construct a new matrix::matrix object.
 *
 * @param _rows
 * @param _cols
 */
matrix::matrix(const uint64_t &_rows, const uint64_t &_cols)
    : rows(_rows), cols(_cols)
{
    if (rows == 0 or cols == 0)
        throw zero_size;
    elements = vector<double>(rows * cols);
}
/**
 * @brief Construct a new matrix::matrix object.
 *
 * @param _rows
 * @param _cols
 * @param _elements
 */
matrix::matrix(const uint64_t &_rows, const uint64_t &_cols, const vector<double> &_elements)
    : rows(_rows), cols(_cols), elements(_elements)
{
    if (rows == 0 or cols == 0)
        throw zero_size;
    if (_elements.size() != rows * cols)
        throw initializer_wrong_size;
}

/**
 * @brief Get number of rows from a matrix.
 *
 * @return Number of rows.
 */
uint64_t matrix::get_rows() const
{
    return rows;
}
/**
 * @brief Get number of columns from a matrix.
 *
 * @return Number of columns.
 */
uint64_t matrix::get_cols() const
{
    return cols;
}
// Overloaded Operators from Lecture Notes.
/**
 * @brief Allows for accessing matrix elements.
 *
 * @param row
 * @param col
 * @return Returns a matrix element.
 */
double &matrix::operator()(const uint64_t &row, const uint64_t &col)
{
    return elements[(cols * row) + col];
}
/**
 * @brief Allows for accessing matrix elements.
 *
 * @param row
 * @param col
 * @return Returns a matrix element.
 */
const double &matrix::operator()(const uint64_t &row, const uint64_t &col) const
{
    return elements[(cols * row) + col];
}
/**
 * @brief Prints a matrix object.
 *
 * @param out
 * @param m
 * @return Prints out matrix.
 */
ostream &operator<<(ostream &out, const matrix &m)
{
    out << '\n';
    for (uint64_t i = 0; i < m.get_rows(); i++)
    {
        out << "( ";
        for (uint64_t j = 0; j < m.get_cols(); j++)
            out << m(i, j) << '\t';
        out << ")\n";
    }
    return out;
}
/**
 * @brief Overloaded operator for matrix addition.
 *
 * @param b
 * @return Returns the resulting matrix from addition.
 */
matrix matrix::operator+(const matrix &b)
{
    if ((rows != b.get_rows()) or (cols != b.get_cols()))
        throw matrix::incompatible_sizes_add;
    matrix c(rows, cols);
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            c(i, j) = elements[i * cols + j] + b(i, j);
    return c;
}
/**
 * @brief Overloaded operator for matrix addition assignment.
 *
 * @param a
 * @param b
 * @return Returns resulting matrix.
 */
matrix operator+=(matrix &a, const matrix &b)
{
    a = a + b;
    return a;
}
/**
 * @brief Overloaded operator for negation of a matrix.
 *
 * @param m
 * @return Returns resulting matrix.
 */
matrix operator-(const matrix &m)
{
    matrix c(m.get_rows(), m.get_cols());
    for (uint64_t i = 0; i < m.get_rows(); i++)
        for (uint64_t j = 0; j < m.get_cols(); j++)
            c(i, j) = -m(i, j);
    return c;
}
/**
 * @brief Overloaded operator for matrix subtraction.
 *
 * @param b
 * @return Returns resulting matrix.
 */
matrix matrix::operator-(const matrix &b)
{
    if ((rows != b.get_rows()) or (cols != b.get_cols()))
        throw matrix::incompatible_sizes_add;
    matrix c(rows, cols);
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            c(i, j) = elements[i * cols + j] - b(i, j);
    return c;
}
/**
 * @brief Overloaded operator for subtraction assignment of matrices.
 *
 * @param a
 * @param b
 * @return Returns resulting matrix.
 */
matrix operator-=(matrix &a, const matrix &b)
{
    a = a - b;
    return a;
}

/**
 * @brief Overloaded operator for matrix multiplication.
 *
 * @param b
 * @return Returns resulting matrix.
 */
matrix matrix::operator*(const matrix &b)
{
    if (cols != b.get_rows())
        throw matrix::incompatible_sizes_multiply;
    matrix c(rows, b.get_cols());
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < b.get_cols(); j++)
            for (uint64_t k = 0; k < cols; k++)
                c(i, j) += elements[i * cols + k] * b(k, j);
    return c;
}
/**
 * @brief Overloaded operator for scalar multiplication.
 *
 * @param s
 * @return Returns resulting matrix.
 */
matrix matrix::operator*(const double &s)
{
    matrix c(rows, cols);
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            c(i, j) = s * elements[i * cols + j];
    return c;
}
/**
 * @brief Overloaded operator for scalar division.
 *
 * @param s
 * @return Returns resulting matrix.
 */
matrix matrix::operator/(const double &s)
{
    matrix c(rows, cols);
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            c(i, j) = elements[i * cols + j] / s;
    return c;
}
/**
 * @brief Overloaded operator for scalar multiplication on the left.
 *
 * @param s
 * @param m
 * @return Returns resulting matrix.
 */
matrix operator*(const double &s, const matrix &m)
{
    matrix c(m.get_rows(), m.get_cols());
    for (uint64_t i = 0; i < m.get_rows(); i++)
        for (uint64_t j = 0; j < m.get_cols(); j++)
            c(i, j) = s * m(i, j);
    return c;
}
/**
 * @brief Overloaded operator for scalar multiplication on the right.
 *
 * @param m
 * @param s
 * @return Returns resulting matrix.
 */
matrix operator*(const matrix &m, const double &s)
{
    return s * m;
}

// End of Matrix Class Functions from Lecture Notes
//  The follow functions were added by me.

/**
 * @brief Calculates the determinant of a matrix. This was adapted from my first project.
 *
 * @param rows
 * @param cols
 * @param m
 * @return Returns the determinant of a matrix as a double.
 */
double determinant(uint64_t rows, uint64_t cols, const matrix &m)
{
    double D = 0;
    int sign = 1;
    uint64_t newrow = 0;
    uint64_t newcol = 0;
    matrix newMat(m.get_rows(), m.get_cols());

    if (rows <= 0)
    {
        return 0; // stop
    }
    if (rows == 1)
        return m(0, 0);

    if (rows == 2) // when row==2 col will also equal 2 since square matrix
    {
        return m(0, 0) * m(1, 1) - m(1, 0) * m(0, 1);
    }

    else
    {
        for (uint64_t k = 0; k < cols; k++) // loop through col
        {
            newrow = 0; // restart column when you move to new row
            // newcol = 0;

            for (uint64_t i = 1; i < rows; i++) // row 0 never enters new matrix
            {
                newcol = 0; // restart column when you move to new row

                for (uint64_t r = 0; r < cols; r++) // loop through the elements in each row
                {
                    if (r != k) // if element is in the current column skip it
                    {
                        newMat(newrow, newcol) = m(i, r); // elements not in same row/col are assigned to new (sub) matrix
                        newcol++;
                    }
                }
                newrow++;
            }

            D += sign * m(0, k) * determinant(rows - 1, cols - 1, newMat); // multiple the kth element in row 0 by sub matrix
            sign = -1 * sign;
        }
    }
    return D;
}

/**
 * @brief Extracts a row from a matrix.
 *
 * @param m
 * @param row
 * @return Returns row matrix.
 */

matrix extract_row(const matrix &m, uint64_t row)
{
    matrix temp(1, m.get_cols()); // Initialize matrix object.
    for (uint64_t i = 0; i < m.get_cols(); i++)
    {
        temp(0, i) = m(row, i); // Obtain the row.
    }
    return temp; // Return
}

/**
 * @brief Extracts a column from a matrix.
 *
 * @param m
 * @param col
 * @return Returns a column matrix.
 */
matrix extract_col(const matrix &m, uint64_t col)
{
    matrix temp(m.get_rows(), 1); // Initialize the matrix object.
    for (uint64_t i = 0; i < m.get_rows(); i++)
    {
        temp(i, 0) = m(i, col); // Obtain the column.
    }
    return temp; // Return
}

/**
 * @brief Cofactor matrix needed for the inverse of a matrix. Maybe I should rewrite this with determinant code.
 * This code was adapted from my determinant code.
 * @param rows
 * @param cols
 * @param m
 * @param p
 * @param q
 * @return Returns a cofactor matrix based on p and q.
 */
matrix cofactor(uint64_t rows, uint64_t cols, const matrix &m, uint64_t p, uint64_t q)
{
    matrix cofact(rows - 1, cols - 1);
    uint64_t newrow = 0;
    uint64_t newcol = 0;

    for (uint64_t i = 0; i < rows; i++) // row 0 never enters new matrix
    {
        newcol = 0; // restart column when you move to new row

        if (i != p)
        {
            for (uint64_t r = 0; r < cols; r++) // loop through the elements in each row
            {
                if (r != q) // if element is in the current column skip it
                {
                    cofact(newrow, newcol) = m(i, r); // elements not in same row/col are assigned to new (sub) matrix
                    newcol++;                         // Move to next column.
                }
            }
            newrow++; // Move to next row
        }
    }

    return cofact;
}
/**
 * @brief Finds the inverse of a matrix.
 * The calculation of the adjoint comes from https://www.geeksforgeeks.org/adjoint-inverse-matrix/ with some modifications.
 * @param m
 * @return Returns the inverse of a matrix.
 */
matrix inverse(const matrix &m)
{

    int64_t sign = 1;
    uint64_t rows = m.get_rows();
    uint64_t cols = m.get_cols();
    matrix cofact(rows, cols);
    matrix adj(rows, cols);

    for (uint64_t i = 0; i < rows; i++)
    {
        for (uint64_t j = 0; j < cols; j++)
        {

            if ((i + j) % 2 == 0)
            {
                sign = 1;
            }
            else
            {
                sign = -1;
            }

            // Interchanging rows and columns to get the transpose of the cofactor matrix
            adj(j, i) = (sign) * (determinant(rows - 1, cols - 1, cofactor(rows, cols, m, i, j)));
        }
    }

    // Then we can get the inverse from adjoint and determinant
    matrix inverse(rows, cols);
    for (uint64_t i = 0; i < rows; i++)
        for (uint64_t j = 0; j < cols; j++)
            inverse(i, j) = adj(i, j) / determinant(rows, cols, m); // Adjoint divided by determinant will provide the inverse
    return inverse;
}

/**
 * @brief Calculates the mean matrix from a matrix.
 *
 * @param rows
 * @param cols
 * @param m
 * @return Returns a mean matrix.
 */
matrix mean(uint64_t rows, uint64_t cols, const matrix &m)
{
    double temp = 0;
    matrix mu(rows, 1); // Initialize mean matrix.
    for (uint64_t j = 0; j < cols; j++)
    {

        for (uint64_t i = 0; i < rows; i++)
        {
            temp += m(i, j); // Add values.
        }
        mu(j, 0) = temp / static_cast<double>(rows); // Divide by the number of observations.
        temp = 0;
    }
    return mu; // Return mean matrix.
}

/**
 * @brief Calculates the covariance matrix from a matrix.
 *
 * @param rows
 * @param cols
 * @param m
 * @return Returns a covariance matrix.
 */
matrix covariance_matrix(uint64_t rows, uint64_t cols, const matrix &m)
{
    double cov = 0;
    double meannum1;
    double meannum2;
    matrix cov_mat(cols, cols);            // Initialize the covariance matrix.
    matrix mean_mat = mean(rows, cols, m); // Obtain means.
    for (uint64_t i = 0; i < cols; i++)
    {
        meannum1 = mean_mat(i, 0);
        for (uint64_t k = 0; k < cols; k++)
        {
            meannum2 = mean_mat(k, 0);
            for (uint64_t j = 0; j < rows; j++)
            {

                cov += (m(j, i) - meannum1) * (m(j, k) - meannum2); // Numerator of covariance of two variables.
            }
            cov_mat(i, k) = cov / (rows - 1); // Divide by the number of observations subtract 1.
            cov = 0;
        }
    }
    return cov_mat;
}

/**
 * @brief Calculates the transpose of a matrix.
 *
 * @param rows
 * @param cols
 * @param m
 * @return Returns the transpose of a matrix.
 */
matrix transpose(uint64_t rows, uint64_t cols, const matrix &m)
{
    matrix transpose_mat(cols, rows); // Initialize the transpose matrix.

    for (uint64_t i = 0; i < rows; i++)
    {
        for (uint64_t j = 0; j < cols; j++)
        {
            transpose_mat(j, i) = m(i, j); // Transpose
        }
    }

    return transpose_mat; // Return
}
// End of Matrix Class Functions.

// Functions for the EMalgo class.
/**
 * @brief Calculates the density of a multivariate normal.
 *
 * @param data
 * @param mu
 * @param cov
 * @return Returns a matrix containing the density of a multivariate normal.
 */
matrix dmvnorm(matrix &data, matrix &mu, matrix &cov)
{
    uint64_t rows = data.get_rows();
    uint64_t cols = data.get_cols();

    matrix density(rows, 1); // Density matrix to return the density of each observation.
    double pival = numbers::pi;

    matrix inv_sigma(cols, cols);
    inv_sigma = inverse(cov);  // Inverse of the covariance matrix.
    matrix ztrans(rows, cols); // Variable - mean of variable.
    matrix z(cols, rows);
    double detval = determinant(cols, cols, cov); // Determinant of the covariance matrix.

    // Loop through the size of data.
    for (uint64_t i = 0; i < rows; i++)
    {
        for (uint64_t j = 0; j < cols; j++)
        {
            ztrans(i, j) = data(i, j) - mu(0, j); // X-mu
        }
    }
    matrix temp(1, cols);
    matrix exp_val(1, cols);
    matrix exp_val2(1, 1);
    matrix returnval(1, 1);
    for (uint64_t i = 0; i < rows; i++) // Loop through the rows to get the density of each observation.
    {

        temp = extract_row(ztrans, i);
        z = transpose(1, cols, temp);
        exp_val = temp * inv_sigma;
        exp_val2 = exp_val * z;
        density(i, 0) = exp(exp_val2(0, 0) * (-0.5)) / (sqrt(pow(2 * pival, cols) * detval));
    }
    return density;
}

/**
 * @brief Creates a zero matrix.
 *
 * @param rows
 * @param cols
 * @return Returns a zero matrix.
 */
matrix zero_matrix(uint64_t rows, uint64_t cols)
{
    matrix zeros(rows, cols);
    for (uint64_t i = 0; i < rows; i++)
    {
        for (uint64_t j = 0; j < cols; j++)
        {
            zeros(i, j) = 0; // Zero matrix
        }
    }
    return zeros;
}

/**
 * @brief Computes the complete data log likelihood of a Gaussian mixture model.
 *
 * @param data
 * @param pigs
 * @param mus
 * @param sigmas
 * @param G
 * @return Returns a matrix containing the complete data log-likelihood.
 */
double compute_log_likelihood(matrix &data, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G)
{
    matrix loglike(data.get_rows(), 1);
    loglike = zero_matrix(data.get_rows(), 1); // Make sure this is zero to begin.
    matrix dens(data.get_rows(), 1);
    for (uint64_t g = 0; g < G; g++)
    {
        dens = dmvnorm(data, mus[g], sigmas[g]) * pigs[g]; // Calculate the density for each group and multiply by probability of each group.
        loglike = loglike + dens;                          // Add the densities for each group.
    }
    // Then need to sum log over n.
    double sum_log_like = 0;

    for (uint64_t i = 0; i < data.get_rows(); i++)
    {
        sum_log_like = sum_log_like + log(loglike(i, 0)); // Take the log of density for each observation.
    }

    return sum_log_like;
}

/**
 * @brief Makes a vector of matrix elements.
 *
 * @param G
 * @param rows
 * @param cols
 * @return Returns a vector of matrices.
 */
vector<matrix> vec_mat(uint64_t G, uint64_t rows, uint64_t cols)
{
    vector<matrix> final_vec;
    for (uint64_t i = 0; i < G; i++) // Loop through the number of groups.
    {
        matrix mat1(rows, cols);
        final_vec.push_back(mat1); // Add a matrix to each vector element.
    }
    return final_vec; // Should return vector of length G of rows x cols matrices.
}

// next E and M steps

/**
 * @brief Function to perform the E step of the EM.
 * Use the expectation of log-likelihood given current model estimates to obtain
 * an estimate of latent variables. These estimates provide a probability of each observation
 * belonging to each group.
 * @param data
 * @param pigs
 * @param mus
 * @param sigmas
 * @param G
 * @return Returns a matrix containing soft classifications.
 */
matrix e_step(matrix &data, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G)
{
    matrix zigs(data.get_rows(), G);
    matrix tempmat(1, G);
    matrix dens(1, 1);
    matrix sum_mat(data.get_rows(), 1);
    sum_mat = zero_matrix(data.get_rows(), 1);

    for (uint64_t i = 0; i < data.get_rows(); i++)
    {
        for (uint64_t g = 0; g < G; g++)
        {
            tempmat = extract_row(data, i);
            dens = dmvnorm(tempmat, mus[g], sigmas[g]) * pigs[g]; // Density at current estimates multiplied by prior information
            zigs(i, g) = dens(0, 0);                              // Numerator of zigs.
        }
    }
    // Obtain row totals.
    for (uint64_t i = 0; i < data.get_rows(); i++)
    {
        for (uint64_t j = 0; j < G; j++)
        {
            sum_mat(i, 0) += zigs(i, j);
        }
    }
    // Divide by each row total.

    for (uint64_t i = 0; i < data.get_rows(); i++)
    {
        for (uint64_t j = 0; j < G; j++)
        {
            zigs(i, j) = zigs(i, j) / sum_mat(i, 0); // Obtaining estimates for Z (probability each observation belongs to each class).
            if (zigs(i, j) < 0.0001)                 // This is just to introduce some numeric stability. Any probability smaller than this is essentially zero.
            {
                zigs(i, j) = 0;
            }
        }
    }

    return zigs;
}

// M step is next
/**
 * @brief Function to perform the M step of the EM algorithm.
 * This function maximizes the log-likelihood to obtain MLE's for model parameters.
 * @param data
 * @param zigs
 * @param pigs
 * @param mus
 * @param sigmas
 * @param G
 * @return Matrices are updated but nothing is returned.
 */

void m_step(matrix &data, matrix &zigs, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G)
{
    // the pigs, mus, sigmas that are read in here should just be 0 matrices
    matrix ngs(G, 1);
    matrix tempcol(data.get_rows(), 1);
    matrix temp(data.get_rows(), data.get_cols());
    ngs = zero_matrix(data.get_rows(), 1);
    matrix sumval(1, 1);
    matrix temprow(1, data.get_cols());
    matrix ztemp(1, data.get_cols());

    // Update pigs: mixing proportions for each group
    sumval = zero_matrix(1, 1);
    for (uint64_t g = 0; g < G; g++)
    {
        for (uint64_t i = 0; i < data.get_rows(); i++)
        {
            ngs(g, 0) += zigs(i, g);
        }
        pigs[g](0, 0) = ngs(g, 0) / data.get_rows();
    }

    // Update mus: means for each group.
    // Both mean and covariance equations are weighted by zigs: the probability of each observations belonging to each group.

    for (uint64_t g = 0; g < G; g++)
    {
        tempcol = extract_col(zigs, g);
        temp = transpose(tempcol.get_rows(), tempcol.get_cols(), tempcol) * data;
        for (uint64_t j = 0; j < data.get_cols(); j++)
        {
            mus[g](0, j) = temp(0, j) / ngs(g, 0);
        }
    }

    // Update sigmas: covariance for each group.

    for (uint64_t g = 0; g < G; g++)
    {
        sigmas[g] = zero_matrix(data.get_cols(), data.get_cols());
        for (uint64_t i = 0; i < data.get_rows(); i++)
        {
            temprow = extract_row(data, i);
            ztemp = (temprow - mus[g]);
            sigmas[g] = sigmas[g] + zigs(i, g) * (transpose(1, data.get_cols(), ztemp) * ztemp) / ngs(g, 0);
        }
    }
}

/**
 * @brief Initializes the probability matrix with soft classifications.
 *
 * @param rows
 * @param G
 * @return Returns random soft classifications.
 */
matrix random_soft(uint64_t rows, uint64_t G)
{
    matrix v(rows, G);
    double row_tot = 0;
    random_device rd;
    mt19937_64 mt(rd());
    uniform_real_distribution<double> urd(0, 1); // As this is a matrix of probabilities all numbers need to be between 0 and 1.

    // Rows represent the observations.
    // Columns represent the groups.
    // Those rows need to add to 1 as this matrix represents the probability of each observations belonging to each group.
    if (G == 2)
    {
        for (uint64_t i = 0; i < rows; i++)
        {

            v(i, 0) = urd(mt);

            v(i, 1) = 1 - v(i, 0); // Ensuring sum to 1.
        }
    }
    else
    {
        for (uint64_t i = 0; i < rows; i++)
        {
            for (uint64_t g = 0; g < G; g++)
            {
                v(i, g) = urd(mt);
                row_tot += v(i, g); // Add row sum
            }
            for (uint64_t g = 0; g < G; g++)
            {
                v(i, g) = v(i, g) / row_tot; // Divide by row sum to ensure the rows sum to 1.
            }
            row_tot = 0;
        }
    }

    return v;
}

/**
 * @brief Function to perform the EM.
 *
 * @param data
 * @param pigs
 * @param mus
 * @param sigmas
 * @param G
 * @return Returns estimated soft classifications.
 */

matrix EM(matrix &data, vector<matrix> &pigs, vector<matrix> &mus, vector<matrix> &sigmas, uint64_t G)
{
    // Iterate between E and M steps until convergence of log-likelihood is achieved.
    uint64_t iter = 0;
    uint64_t maxiter = 1000; // Maximum number of iterations in the event that convergence does not occur.
    double eps = 0.000001;   // Tolerance level for convergence of log-likelihood.
    vector<double> logval;
    logval.push_back(0); // To ensure the algorithm begins
    matrix zigs(data.get_rows(), G);
    zigs = random_soft(data.get_rows(), G); // Inital group memberships

    for (uint64_t g = 0; g < G; g++) // Initialize everything to zero
    {
        mus[g] = zero_matrix(1, data.get_cols());
        sigmas[g] = zero_matrix(data.get_cols(), data.get_cols());
        pigs[g] = zero_matrix(1, 1);
    }

    while (iter < maxiter && !isnan(logval[iter]))
    {
        if (iter == 0)
        {
            // Need our first M step before entering the E and M iterations as an initial means, sigmas, and pigs are needed for E step.

            m_step(data, zigs, pigs, mus, sigmas, G);
            iter += 1;
            logval.push_back(compute_log_likelihood(data, pigs, mus, sigmas, G));
            // iter += 1;
        }
        else
        { // Now iterate between E and M until convergence of likelihood occurs or max iterations are achieved.
            // Estep
            zigs = e_step(data, pigs, mus, sigmas, G);
            // Set everything to zero again before the m step and use the zigs just estimated.
            for (uint64_t g = 0; g < G; g++) // Initialize everything to zero.
            {
                mus[g] = zero_matrix(1, data.get_cols());
                sigmas[g] = zero_matrix(data.get_cols(), data.get_cols());
                pigs[g] = zero_matrix(1, 1);
            }
            // Mstep
            m_step(data, zigs, pigs, mus, sigmas, G);
            iter += 1;                                                            // Increase number of iterations to allow for while loop to continue.
            logval.push_back(compute_log_likelihood(data, pigs, mus, sigmas, G)); // Compute log-likelihood

            if ((iter >= 2) && abs(logval[iter] - logval[iter - 1]) <= eps) // Check convergence
            {
                break; // Break if converged.
            }
        }
    }

    return zigs; // Return estimated memberships.
}
/**
 * @brief Function to transform the soft classifications (zigs) to hard classifications.
 *
 * @param zigs
 * @return Returns hard classifications as a matrix.
 */
matrix classify(matrix &zigs)
{
    matrix classified(zigs.get_rows(), 1);
    double max = 0;
    uint64_t index = 0;

    for (uint64_t i = 0; i < zigs.get_rows(); i++)
    {
        for (uint64_t j = 0; j < zigs.get_cols(); j++)
        {

            if (zigs(i, j) > max)
            {
                max = zigs(i, j); // The group with the highest probability gets assigned to the observation.
                index = j;
            }
        }
        classified(i, 0) = index;
        index = 0;

        max = 0;
    }

    return classified;
}

/**
 * @brief Code to calculate binomial coefficients.
 * This function comes from https://www.geeksforgeeks.org/binomial-coefficient-dp-9/
 * @param n
 * @param k
 * @return Returns the binomial coefficient as a double.
 */
double binomial_coeff(double n, double k)
{
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;

    return binomial_coeff(n - 1, k - 1) + binomial_coeff(n - 1, k);
}

/**
 * @brief Function to calculate the adjusted Rand index.
 *
 * @param classified
 * @param true_class
 * @return Returns ARI as a double.
 */
double ARI(matrix &classified, matrix &true_class)
{
    // Count the number of unique values in classified to get estimated number of groups.
    vector<double> classified_vec;
    vector<double> sorted_classified_vec;
    for (uint64_t i = 0; i < classified.get_rows(); i++)
    {
        classified_vec.push_back(classified(i, 0));
        sorted_classified_vec.push_back(classified(i, 0));
    }

    // The counting of unique values
    sort(sorted_classified_vec.begin(), sorted_classified_vec.end()); // Need a sorted vector to count unique entries, however we want to calculate ARI on unsorted vector

    uint64_t uniqueCount = 1;
    for (uint64_t s = 1; s <= classified.get_rows(); s++)
    {
        if (sorted_classified_vec[s] - sorted_classified_vec[s - 1] != 0)
        {
            uniqueCount += 1;
        }
    }

    // Count the unique values int the true vector to get the true number of groups.
    vector<double> true_vec;
    vector<double> sorted_true_vec;
    for (uint64_t i = 0; i < true_class.get_rows(); i++)
    {
        true_vec.push_back(true_class(i, 0));
        sorted_true_vec.push_back(true_class(i, 0));
    }
    sort(sorted_true_vec.begin(), sorted_true_vec.end()); // Need a sorted vector to count unique entries, however we want to calculate ARI on unsorted vector
    uint64_t uniqueCount2 = 1;
    for (uint64_t s = 1; s <= classified.get_rows(); s++)
    {
        if (sorted_true_vec[s] - sorted_true_vec[s - 1] != 0)
        {
            uniqueCount2 += 1;
        }
    }

    // Need to obtain the contingency table in order to calculate ARI.
    matrix count_val(uniqueCount, uniqueCount2);
    count_val = zero_matrix(uniqueCount, uniqueCount2);
    for (uint64_t i = 0; i < uniqueCount; i++)
    {
        for (uint64_t j = 0; j < uniqueCount2; j++)
        {
            for (uint64_t r = 0; r < classified.get_rows(); r++)
            {
                if (classified_vec[r] == i && true_vec[r] == j) // Counting the number of times the vectors have each combination of values.
                {
                    count_val(i, j) = count_val(i, j) + 1.0;
                }
            }
        }
    }
    // count_val is our contingency table without totals.
    // Now we need row and column totals.
    matrix cluster_tot(uniqueCount, 1);
    matrix true_totals(uniqueCount2, 1);
    double tempval = 0;

    for (uint64_t i = 0; i < uniqueCount; i++)
    {
        tempval = 0;
        for (uint64_t j = 0; j < uniqueCount2; j++)
        {
            tempval += count_val(i, j);
        }
        cluster_tot(i, 0) = tempval; // Row totals
    }

    for (uint64_t i = 0; i < uniqueCount2; i++)
    {
        tempval = 0;
        for (uint64_t j = 0; j < uniqueCount; j++)
        {
            tempval += count_val(i, j);
        }
        true_totals(i, 0) = tempval; // Column totals.
    }
    // Now need to calculate ARI using binomial coefficient function.
    double sum_nij = 0;
    double sum_ai = 0;
    double sum_bj = 0;
    double n;

    n = binomial_coeff((double)classified.get_rows(), 2);

    for (uint64_t i = 0; i < uniqueCount; i++)
    {
        sum_ai += binomial_coeff(cluster_tot(i, 0), 2);
        for (uint64_t j = 0; j < uniqueCount2; j++)
        {
            sum_nij += binomial_coeff(count_val(i, j), 2);
            if (i == uniqueCount - 1)
            {
                sum_bj += binomial_coeff(true_totals(j, 0), 2);
            }
        }
    }

    double ARI;
    // Now calculate ARI
    ARI = (sum_nij - (sum_ai * sum_bj) / n) / (0.5 * (sum_ai + sum_bj) - (sum_ai * sum_bj) / n);

    return ARI; // Return ARI.
}

// End of Functions for EMalgo class.

// Main function and function to read in data
/**
 * @brief Function to read in comma separated data.
 *
 * @param in
 * @param tot
 * @return Returns a vector of doubles containing read in information.
 */
vector<double> read_numbers(const string &in, uint64_t &tot)
{
    vector<double> v;
    string s;
    istringstream string_stream(in);
    try
    {
        while (getline(string_stream, s, ',')) // Read string containing commas into a vector.
        {

            v.push_back(stod(s));
            tot++;
        }
    }
    catch (const invalid_argument &e) // Error checking comes from lecture notes.
    {
        throw invalid_argument("Expected a number!");
    }
    catch (const out_of_range &e) // Error checking comes from lecture notes.
    {
        throw out_of_range("Number is out of range!");
    }
    return v;
}

/**
 * @brief Main function to run the program.
 *
 * @param argc
 * @param argv
 * @return int
 */
int main(int argc, char *argv[])
{

    // Check that the correct number of inputs are provided.
    if (argc != 4 && argc != 5 && argc != 1) // Four or five depending on whether true groupings are provided.
    {
        printf("Error: you have not provided the correct number of inputs.\n");
        return -1;
    }
    // If no arguments are provided print out program information.
    if (argc == 1)
    {
        ifstream input("text.txt");
        if (!input.is_open()) // Error checking comes from lecture notes.
        {
            cout << "Error opening file!";
            return -1;
        }

        char c;
        while (input.get(c))
            cout << c;

        input.close();
        return -1;
    }
    // Read in G and input data
    string line;
    string output;
    string filename = argv[1];
    ifstream file(filename);
    vector<double> datav;
    uint64_t rows = 0;
    uint64_t tots = 0;
    // Check if the string is numeric
    string group_num = argv[2];
    for (std::vector<int>::size_type i = 0; i < group_num.size(); i++) // I'm pretty sure this conversion came from online.
    {
        if (!isdigit(group_num[i]))
        {
            printf("G needs to be a positive integer.\n");
            exit(0);
        }
    }

    // Convert to uint64_t type
    uint64_t G;
    std::istringstream iss(argv[2]);
    iss >> G;

    // Read in data, this will occur in two steps.
    // Read data into a string separating all values by commas.
    // Then read_numbers can be used to read string into a vector based on commas.
    uint64_t i = 0;
    if (file.is_open())
    {
        while (getline(file, line))
        { // Get line up until whitespace.

            if (i == 0) // Concatenation depends on which rows in the data it is.
            {
                output = output + line;
                i++;
            }
            else
            {
                output = output + ',' + line; // Separate by comma.
            }
            rows++;
        }

        try
        {
            datav = read_numbers(output, tots); // Read comma deliminated string into vector.
        }
        catch (const invalid_argument &e) // Error checking comes from lecture notes.
        {
            cout << "Error: " << e.what() << '\n';
            exit(0);
        }
        catch (const out_of_range &e) // Error checking comes from lecture notes.
        {
            cout << "Error: " << e.what() << '\n';
            exit(0);
        }
    }
    file.close();
    // Should close the file?

    try
    {
        matrix data2(rows, tots / rows, datav); // Will be read in regardless of whether argc==4 or 5

        // Initialize values for the EM.
        vector<matrix> pigs;
        vector<matrix> mus;
        vector<matrix> sigmas;

        // EM set up
        for (uint64_t g = 0; g < G; g++)
        {
            // pigs
            pigs = vec_mat(G, 1, 1);
            pigs[g] = matrix(1, 1, {0});
            // cout << pigs[1];

            // mus
            mus = vec_mat(G, 1, tots / rows); // G,1,p
            mus[g] = matrix(1, tots / rows, {0, 0});
            // cout << mus[1];

            // sigmas
            sigmas = vec_mat(G, tots / rows, tots / rows); // G, p, p
            sigmas[g] = matrix(tots / rows, tots / rows, {0, 0, 0, 0});
            // cout << sigmas[0];
        }

        printf("EM Algorithm Results:\n");
        printf("zigs:\n");
        matrix zigs(rows, G);
        zigs = EM(data2, pigs, mus, sigmas, G); // Run the EM
        cout << zigs;                           // Print out predicted soft classifications.

        for (uint64_t g = 0; g < G; g++) // Print out the model parameters by group.
        {
            printf("Group ");
            printf("%" PRIu64 " \n", g + 1);
            printf("Mean vector:\n");
            cout << mus[g];
            printf("Mixing proportion:\n");
            cout << pigs[g];
            printf("Covariance matrix:\n");
            cout << sigmas[g];
        }

        // printf("Classified:\n");
        matrix classified(data2.get_rows(), 1);
        classified = classify(zigs); // Soft to hard classifications.
        // cout << classified;

        // Below code prints to output
        vector<double> classified_vev;
        for (uint64_t p = 0; p < data2.get_rows(); p++)
        {
            classified_vev.push_back(classified(p, 0));
        }
        string filename3 = argv[4];
        ofstream output3(filename3);
        if (!output3.is_open()) // Error checking comes from lecture notes.
        {
            cout << "Error opening output file!";
            return -1;
        }

        for (uint64_t m = 0; m < data2.get_rows(); m++)
        {
            output3 << classified_vev[m] << '\n';
        }
        output3.close();

        // Below code is if true classes are provided.
        if (argc == 5) // If true class data is provided then we can calculate ARI.
        {
            string line2;
            string output2;
            string filename2 = argv[3];
            ifstream file2(filename2);
            vector<double> truedata;

            // Since this is a 1D vector we don't need to worry about commas.
            // Just put into a vector based on whitespace.

            try
            {
                while (getline(file2, line2))
                {

                    truedata.push_back(stod(line2));
                }
            }
            catch (const invalid_argument &e) // Error checking comes from lecture notes.
            {
                throw invalid_argument("Expected a number!");
            }
            catch (const out_of_range &e) // Error checking comes from lecture notes.
            {
                throw out_of_range("Number is out of range!");
            }
            file2.close();
            matrix true_class(rows, 1, truedata);
            printf("ARI: %f\n", ARI(classified, true_class)); // Calculate ARI to compare estimated and true classes.
        }
    }
    catch (const exception &e) // Error checking comes from lecture notes.
    {
        cout << "Error: " << e.what() << '\n';
    }
}
