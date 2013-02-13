//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

  *: Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

  *: Neither the name of the University of Wisconsin - Madison nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
//---------------------------------------------------------------------------//
/*!
 * \file MCLS_MatrixTraits.hpp
 * \author Stuart R. Slattery
 * \brief Matrix traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MATRIXTRAITS_HPP
#define MCLS_MATRIXTRAITS_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedMatrixTraits
 * \brief Class for undefined matrix traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Vector, class Matrix>
struct UndefinedMatrixTraits
{
    static inline void notDefined()
    {
	return Matrix::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class MatrixTraits
 * \brief Traits for matrices.
 *
 * MatrixTraits defines an interface for parallel distributed
 * matrices. (e.g. Tpetra::CrsMatrix or Epetra_VbrMatrix).
 */
template<class Vector, class Matrix>
class MatrixTraits
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                       vector_type;
    typedef Matrix                                       matrix_type;
    typedef typename Vector::scalar_type                 scalar_type;
    typedef typename Vector::local_ordinal_type          local_ordinal_type;
    typedef typename Vector::global_ordinal_type         global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty matrix from a
     * given matrix to give the new matrix the same parallel distribution as
     * the matrix parallel row distribution.
     */
    static Teuchos::RCP<Matrix> clone( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel row distribution.
     */
    static Teuchos::RCP<Vector> 
    cloneVectorFromMatrixRows( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel column distribution.
     */
    static Teuchos::RCP<Vector> 
    cloneVectorFromMatrixCols( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the communicator.
     */
    static Teuchos::RCP<const Teuchos::Comm<int> >
    getComm( const Matrix& matrix )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the global number of rows.
     */
    static global_ordinal_type getGlobalNumRows( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Get the local number of rows.
     */
    static local_ordinal_type getLocalNumRows( const Matrix& matrix )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined();
	return 0; 
    }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static global_ordinal_type getGlobalMaxNumRowEntries( const Matrix& matrix )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static global_ordinal_type getGlobalRow( const Matrix& matrix, 
					     const local_ordinal_type& local_row )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static local_ordinal_type getLocalRow( const Matrix& matrix,
					   const global_ordinal_type& global_row )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static global_ordinal_type getGlobalCol( const Matrix& matrix,
					     const local_ordinal_type& local_col )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static local_ordinal_type getLocalCol( const Matrix& matrix,
					   const global_ordinal_type& global_col )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Get the owning process rank for the given global rows.
     */
    static void getGlobalRowRanks( 
	const Matrix& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_rows,
	const Teuchos::ArrayView<int>& ranks )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const Matrix& matrix,
			     const global_ordinal_type& global_row )
    {
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return false; 
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const Matrix& matrix,
			    const local_ordinal_type& local_row )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return false; 
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const Matrix& matrix,
			     const global_ordinal_type& global_col )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined();
	return false; 
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const Matrix& matrix, 
			    const local_ordinal_type& local_col )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return false;
    }

    /*!
     * \brief Get a copy of a global row.
     */
    static void getGlobalRowCopy( 
	const Matrix& matrix,
	const global_ordinal_type& global_row, 
	const Teuchos::ArrayView<global_ordinal_type>& indices,
	const Teuchos::ArrayView<scalar_type>& values,
	std::size_t& num_entries )
    { UndefinedMatrixTraits<Vector,Matrix>::notDefined(); }

    /*!
     * \brief Get a copy of a local row.
     */
    static void getLocalRowCopy( 
	const Matrix& matrix,
	const local_ordinal_type& local_row, 
	const Teuchos::ArrayView<local_ordinal_type>& indices,
	const Teuchos::ArrayView<scalar_type>& values,
	std::size_t& num_entries )
    { UndefinedMatrixTraits<Vector,Matrix>::notDefined(); }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const Matrix& matrix, Vector& vector )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const Matrix& A, const Vector& x, Vector& y )
    { UndefinedMatrixTraits<Vector,Matrix>::notDefined(); }

    /*!
     * \brief Matrix-Matrix multiply C = A*B
     */
    static void multiply( const Teuchos::RCP<const Matrix>& A, 
			  const Teuchos::RCP<const Matrix>& B, 
			  const Teuchos::RCP<Matrix>& C )
    { UndefinedMatrixTraits<Vector,Matrix>::notDefined(); }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<Matrix> copyTranspose( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a copy
     * of a specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<Matrix> 
    copyNearestNeighbors( const Matrix& matrix,
			  const global_ordinal_type& num_neighbors )
    { 
	UndefinedMatrixTraits<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

  };

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MATRIXTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_MatrixTraits.hpp
// ---------------------------------------------------------------------------//

