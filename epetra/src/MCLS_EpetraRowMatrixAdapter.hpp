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
 * \file MCLS_EpetraRowMatrixAdapter.hpp
 * \author Stuart R. Slattery
 * \brief Epetra::RowMatrix Adapter.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_EPETRAROWMATRIXADAPTER_HPP
#define MCLS_EPETRAROWMATRIXADAPTER_HPP

#include <algorithm>

#include <MCLS_DBC.hpp>
#include <MCLS_MatrixTraits.hpp>
#include <MCLS_EpetraHelpers.hpp>

#include <Teuchos_as.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_RowMatrixTransposer.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_VbrMatrix.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MatrixTraits
 * \brief Traits specialization for Epetra_RowMatrix.
 */
template<>
class MatrixTraits<double,int,int,Epetra_Vector,Epetra_RowMatrix>
{
  public:

    //@{
    //! Typedefs.
    typedef typename Epetra_RowMatrix                     matrix_type;
    typedef typename Epetra_Vector                        vector_type;
    typedef typename double                               scalar_type;
    typedef typename int                                  local_ordinal_type;
    typedef typename int                                  global_ordinal_type;
    typedef EpetraMatrixHelpers<matrix_type>              EMH;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel row distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixRows( const matrix_type& matrix )
    { 
	return Epetra::createVector<Scalar,LO,GO>( matrix.getRowMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel column distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixCols( const matrix_type& matrix )
    { 
	Require( matrix.isFillComplete() );
	return Epetra::createVector<Scalar,LO,GO>( matrix.getColMap() );
    }

    /*!
     * \brief Get the communicator.
     */
    static const Teuchos::RCP<const Teuchos::Comm<int> >&
    getComm( const matrix_type& matrix )
    {
	return matrix.getComm();
    }

    /*!
     * \brief Get the global number of rows.
     */
    static GO getGlobalNumRows( const matrix_type& matrix )
    { 
	return Teuchos::as<GO>( matrix.getGlobalNumRows() );
    }

    /*!
     * \brief Get the local number of rows.
     */
    static LO getLocalNumRows( const matrix_type& matrix )
    {
	return Teuchos::as<LO>( matrix.getRowMap()->getNodeNumElements() );
    }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static GO getGlobalMaxNumRowEntries( const matrix_type& matrix )
    {
	return Teuchos::as<GO>( matrix.getGlobalMaxNumRowEntries() );
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static GO getGlobalRow( const matrix_type& matrix, const LO& local_row )
    { 
	Require( matrix.getRowMap()->isNodeLocalElement( local_row ) );
	return matrix.getRowMap()->getGlobalElement( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static LO getLocalRow( const matrix_type& matrix, const GO& global_row )
    { 
	Require( matrix.getRowMap()->isNodeGlobalElement( global_row ) );
	return matrix.getRowMap()->getLocalElement( global_row );
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static GO getGlobalCol( const matrix_type& matrix, const LO& local_col )
    {
	Require( matrix.isFillComplete() );
	Require( matrix.getColMap()->isNodeLocalElement( local_col ) );
	return matrix.getColMap()->getGlobalElement( local_col );
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static LO getLocalCol( const matrix_type& matrix, const GO& global_col )
    {
	Require( matrix.isFillComplete() );
	Require( matrix.getColMap()->isNodeGlobalElement( global_col ) );
	return matrix.getColMap()->getLocalElement( global_col );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const matrix_type& matrix, const GO& global_row )
    {
	return matrix.getRowMap()->isNodeGlobalElement( global_row );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const matrix_type& matrix, const LO& local_row )
    { 
	return matrix.getRowMap()->isNodeLocalElement( local_row );
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const matrix_type& matrix, const GO& global_col )
    { 
	Require( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeGlobalElement( global_col );
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const matrix_type& matrix, const LO& local_col )
    { 
	Require( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeLocalElement( local_col );
    }

    /*!
     * \brief Get a copy of a global row.
     */
    static void getGlobalRowCopy( const matrix_type& matrix,
				  const GO& global_row, 
				  const Teuchos::ArrayView<GO>& indices,
				  const Teuchos::ArrayView<Scalar>& values,
				  std::size_t& num_entries )
    {
	Require( !matrix.isFillComplete() );
	Require( matrix.getRowMap()->isNodeGlobalElement( global_row ) );
	matrix.getGlobalRowCopy( global_row, indices, values, num_entries );
    }

    /*!
     * \brief Get a copy of a local row.
     */
    static void getLocalRowCopy( const matrix_type& matrix,
				 const LO& local_row, 
				 const Teuchos::ArrayView<LO>& indices,
				 const Teuchos::ArrayView<Scalar>& values,
				 std::size_t& num_entries )
    {
	Require( matrix.isFillComplete() );
	Require( matrix.getRowMap()->isNodeLocalElement( local_row ) );
	matrix.getLocalRowCopy( local_row, indices, values, num_entries );
    }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const matrix_type& matrix, 
				  vector_type& vector )
    { 
	matrix.getLocalDiagCopy( vector );
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const matrix_type& A, 
		       const vector_type& x, 
		       vector_type& y )
    {
	A.apply( x, y );
    }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<matrix_type> copyTranspose( const matrix_type& matrix )
    { 
	Epetra::RowMatrixTransposer<Scalar,LO,GO> transposer( matrix );
	return transposer.createTranspose();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class MatrixTraits
 * \brief Traits specialization for Epetra_CrsMatrix.
 */
template<>
class MatrixTraits<double,int,int,Epetra_Vector,Epetra_CrsMatrix>
{
  public:

    //@{
    //! Typedefs.
    typedef typename Epetra_CrsMatrix                     matrix_type;
    typedef typename Epetra_Vector                        vector_type;
    typedef typename double                               scalar_type;
    typedef typename int                                  local_ordinal_type;
    typedef typename int                                  global_ordinal_type;
    typedef EpetraMatrixHelpers<matrix_type>              EMH;
    //@}

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global crss.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const GO& num_neighbors )
    { 
	return EMH::copyNearestNeighbors( matrix, num_neighbors );
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class MatrixTraits
 * \brief Traits specialization for Epetra_VbrMatrix.
 */
template<>
class MatrixTraits<double,int,int,Epetra_Vector,Epetra_VbrMatrix>
{
  public:

    //@{
    //! Typedefs.
    typedef typename Epetra_VbrMatrix                     matrix_type;
    typedef typename Epetra_Vector                        vector_type;
    typedef typename double                               scalar_type;
    typedef typename int                                  local_ordinal_type;
    typedef typename int                                  global_ordinal_type;
    typedef EpetraMatrixHelpers<matrix_type>              EMH;
    //@}

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global vbrs.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const GO& num_neighbors )
    { 
	return EMH::copyNearestNeighbors( matrix, num_neighbors );
    }
};

//---------------------------------------------------------------------------//

#endif // end MCLS_EPETRAROWMATRIXADAPTER_HPP

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraRowMatrixAdapter.hpp
//---------------------------------------------------------------------------//
