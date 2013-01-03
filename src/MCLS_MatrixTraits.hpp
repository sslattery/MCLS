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

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedMatrixTraits
 * \brief Class for undefined matrix traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Scalar, class LO, class GO, class Vector, class Matrix>
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
template<class Scalar, class LO, class GO, class Vector, class Matrix>
class MatrixTraits
{
  public:

    //@{
    //! Typedefs.
    typedef Scalar                                  scalar_type;
    typedef LO                                      local_ordinal_type;
    typedef GO                                      global_ordinal_type;
    typedef Vector                                  vector_type;
    typedef Matrix                                  matrix_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty matrix with
     * the same parallel distribution as the given matrix.
     */
    static Teuchos::RCP<Matrix> clone( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
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
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
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
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the communicator.
     */
    static const Teuchos::RCP<const Teuchos::Comm<int> >&
    getComm( const Matrix& matrix )
    {
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Get the global number of rows.
     */
    static GO getGlobalNumRows( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Get the local number of rows.
     */
    static LO getLocalNumRows( const Matrix& matrix )
    {
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined();
	return 0; 
    }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static GO getGlobalMaxNumRowEntries( const Matrix& matrix )
    {
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static GO getGlobalRow( const Matrix& matrix, const LO& local_ordinal )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static LO getLocalRow( const Matrix& matrix, const GO& global_ordinal )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static GO getGlobalCol( const Matrix& matrix, const LO& local_ordinal )
    {
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static LO getLocalCol( const Matrix& matrix, const GO& global_ordinal )
    {
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return 0; 
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const Matrix& matrix, const GO& global_ordinal )
    {
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return false; 
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const Matrix& matrix, const LO& local_ordinal )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return false; 
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const Matrix& matrix, const GO& global_ordinal )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined();
	return false; 
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const Matrix& matrix, const LO& local_ordinal )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return false; 
    }

    /*!
     * \brief Get a view of a global row.
     */
    static void getGlobalRowView( const Matrix& matrix,
				  const GO& global_ordinal, 
				  Teuchos::ArrayView<const GO> &indices,
				  Teuchos::ArrayView<const Scalar> &values )
    { UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); }

    /*!
     * \brief Get a view of a local row.
     */
    static void getLocalRowView( const Matrix& matrix,
				 const LO& local_ordinal, 
				 Teuchos::ArrayView<const LO> &indices,
				 Teuchos::ArrayView<const Scalar> &values )
    { UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const Matrix& matrix, Vector& vector )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const Matrix& A, const Vector& x, const Vector& y )
    { UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<Matrix> copyNearestNeighbors( const Matrix& matrix,
						      const GO& num_neighbors )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }

    /*!
     * \brief Create a reference-counted pointer to a new matrix by
     * subtracting the transpose of a matrix from the identity matrix. 
     * H = I - A.
     */
    static Teuchos::RCP<Matrix>
    subtractTransposeFromIdentity( const Matrix& matrix )
    { 
	UndefinedMatrixTraits<Scalar,LO,GO,Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MATRIXTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_MatrixTraits.hpp
// ---------------------------------------------------------------------------//

