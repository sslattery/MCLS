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
#include <Teuchos_Comm.hpp>
#include <Teuchos_DefaultMpiComm.hpp>

#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_RowMatrixTransposer.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_VbrMatrix.h>
#include <Epetra_MpiComm.h>

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
	return Teuchos::rcp( new vector_type( matrix.RowMatrixRowMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel column distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixCols( const matrix_type& matrix )
    { 
	Require( matrix.Filled() );
	return Teuchos::rcp( new vector_type( matrix.RowMatrixColMap() );
    }

    /*!
     * \brief Get the communicator.
     */
    static const Teuchos::RCP<const Teuchos::Comm<int> >&
    getComm( const matrix_type& matrix )
    {
#ifdef HAVE_MPI
	Epetra_MpiComm epetra_comm( Teuchos::as<Epetra_MpiComm>( matrix.Comm() ) );
	return Teuchos::rcp( new Teuchos::MpiComm( epetra_comm.getMpiComm() )) ;
#else
	return Teuchos::rcp( new Teuchos::SerialComm<int>() );
#endif
    }

    /*!
     * \brief Get the global number of rows.
     */
    static GO getGlobalNumRows( const matrix_type& matrix )
    { 
	return Teuchos::as<GO>( matrix.NumGlobalRows() );
    }

    /*!
     * \brief Get the local number of rows.
     */
    static LO getLocalNumRows( const matrix_type& matrix )
    {
	return Teuchos::as<LO>( matrix.NumMyRows() );
    }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static GO getGlobalMaxNumRowEntries( const matrix_type& matrix )
    {
	return Teuchos::as<GO>( matrix.MaxNumEntries() );
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static GO getGlobalRow( const matrix_type& matrix, const LO& local_row )
    { 
	Require( matrix.RowMatrixRowMap()->MyLID( local_row ) );
	return matrix.RowMatrixRowMap()->GID( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static LO getLocalRow( const matrix_type& matrix, const GO& global_row )
    { 
	Require( matrix.RowMatrixRowMap()->MyGID( global_row ) );
	return matrix.RowMatrxRowMap()->LID( global_row );
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static GO getGlobalCol( const matrix_type& matrix, const LO& local_col )
    {
	Require( matrix.Filled() );
	Require( matrix.RowMatrixColMap()->myLID( local_col ) );
	return matrix.RowMatrixColMap()->GID( local_col );
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static LO getLocalCol( const matrix_type& matrix, const GO& global_col )
    {
	Require( matrix.Filled() );
	Require( matrix.RowMatrixColMap()->MyGID( global_col ) );
	return matrix.RowMatrixColMap()->LID( global_col );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const matrix_type& matrix, const GO& global_row )
    {
	return matrix.RowMatrixRowMap()->MyGID( global_row );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const matrix_type& matrix, const LO& local_row )
    { 
	return matrix.RowMatrixRowMap()->MyLID( local_row );
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const matrix_type& matrix, const GO& global_col )
    { 
	Require( matrix.Filled() );
	return matrix.RowMatrixColMap()->MyGID( global_col );
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const matrix_type& matrix, const LO& local_col )
    { 
	Require( matrix.Filled() );
	return matrix.RowMatrixColMap()->MyLID( local_col );
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
	Require( !matrix.Filled() );
	Require( matrix.RowMatrixRowMap()->GID( global_row ) );
	LO local_row = matrix.LID( global_row );
	matrix.ExtractMyRowCopy( local_row, 
				 Teuchos::as<LO>(values.size()), 
				 Teuchos::as<LO>(num_entries),
				 values.getRawPtr(), indices.getRawPtr() );
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
	Require( !matrix.Filled() );
	Require( matrix.RowMatrixRowMap()->MyLID( local_row ) );
	matrix.ExtractMyRowCopy( local_row, 
				 Teuchos::as<LO>(values.size()), 
				 Teuchos::as<LO>(num_entries),
				 values.getRawPtr(), indices.getRawPtr() );
    }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const matrix_type& matrix, 
				  vector_type& vector )
    { 
	matrix.ExtractDiagonalCopy( vector );
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const matrix_type& A, 
		       const vector_type& x, 
		       vector_type& y )
    {
	A.apply( false, x, y );
    }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<matrix_type> copyTranspose( const matrix_type& matrix )
    { 
	Epetra_RowMatrixTransposer transposer( &matrix );

	Teuchos::RCP<Epetra_CrsMatrix> tranpose_matrix = Teuchos::rcp(
	    new Epetra_CrsMatrix( Copy, matrix.RowMatrixRowMap(), 0 ) );

	transposer.CreateTranspose( true, tranpose_matrix.getRawPtr() );

	return tranpose_matrix;
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
