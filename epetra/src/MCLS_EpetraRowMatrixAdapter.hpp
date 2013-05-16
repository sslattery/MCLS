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
#include "MCLS_EpetraHelpers.hpp"

#include <Teuchos_as.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_OpaqueWrapper.hpp>

#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>
#include <Epetra_MpiDistributor.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_Import.h>

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#include <Epetra_MpiComm.h>
#endif

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MatrixTraits
 * \brief Traits specialization for Epetra_RowMatrix.
 */
template<>
class MatrixTraits<Epetra_Vector,Epetra_RowMatrix>
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_RowMatrix                              matrix_type;
    typedef Epetra_Vector                                 vector_type;
    typedef double                                        scalar_type;
    typedef int                                           local_ordinal_type;
    typedef int                                           global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty matrix from a
     * given matrix to give the new matrix the same parallel distribution as
     * the matrix parallel row distribution. We're making a CrsMatrix here
     * but we need this for matrix-matrix multiply operations which require a
     * CrsMatrix anyway.
     */
    static Teuchos::RCP<matrix_type> clone( const matrix_type& matrix )
    { 
	return Teuchos::rcp( 
	    new Epetra_CrsMatrix(Copy,matrix.RowMatrixRowMap(),0) );
    }

    /*!
     * \brief Create a reference-counted pointer to a new matrix filled with
     * values imported from a given matrix with a parallel distribution as
     * given by the input rows.
     */
    static Teuchos::RCP<matrix_type> exportFromRows( 
        const matrix_type& matrix,
        const Teuchos::ArrayView<const global_ordinal_type>& global_rows )
    { 
	Epetra_Map target_map( -1, 
                               Teuchos::as<int>(global_rows.size()),
                               global_rows.getRawPtr(),
                               0,
                               matrix.Comm() );

        Teuchos::RCP<Epetra_CrsMatrix> new_matrix = Teuchos::rcp( 
            new Epetra_CrsMatrix(Copy,target_map,0) );

        Epetra_Import importer( matrix.RowMatrixRowMap(), target_map );

        new_matrix->Import( matrix, importer, Insert );

        new_matrix->FillComplete();
        MCLS_ENSURE( new_matrix->Filled() );
        return new_matrix;        
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel row distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixRows( const matrix_type& matrix )
    { 
	return Teuchos::rcp( new vector_type( matrix.RowMatrixRowMap() ) );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel column distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixCols( const matrix_type& matrix )
    { 
	MCLS_REQUIRE( matrix.Filled() );
	return Teuchos::rcp( new vector_type( matrix.RowMatrixColMap() ) );
    }

    /*!
     * \brief Get the communicator.
     */
    static Teuchos::RCP<const Teuchos::Comm<int> >
    getComm( const matrix_type& matrix )
    {
#ifdef HAVE_MPI
	Teuchos::RCP<const Epetra_Comm> epetra_comm = 
	    Teuchos::rcp( &matrix.Comm(), false );
	Teuchos::RCP<const Epetra_MpiComm> mpi_epetra_comm =
	    Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>( epetra_comm );
	Teuchos::RCP<const Teuchos::OpaqueWrapper<MPI_Comm> >
	    raw_mpi_comm = Teuchos::opaqueWrapper( mpi_epetra_comm->Comm() );
	Teuchos::RCP<const Teuchos::MpiComm<int> > teuchos_comm =
	    Teuchos::rcp( new Teuchos::MpiComm<int>( raw_mpi_comm ) );
	return Teuchos::rcp_dynamic_cast<const Teuchos::Comm<int> >(teuchos_comm);
#else
	return Teuchos::rcp( new Teuchos::SerialComm<int>() );
#endif
    }

    /*!
     * \brief Get the global number of rows.
     */
    static global_ordinal_type getGlobalNumRows( const matrix_type& matrix )
    { 
	return Teuchos::as<global_ordinal_type>( matrix.NumGlobalRows() );
    }

    /*!
     * \brief Get the local number of rows.
     */
    static local_ordinal_type getLocalNumRows( const matrix_type& matrix )
    {
	return Teuchos::as<local_ordinal_type>( matrix.NumMyRows() );
    }

    /*!
     * \brief Get the local number of cols.
     */
    static local_ordinal_type getLocalNumCols( const matrix_type& matrix )
    {
	return Teuchos::as<local_ordinal_type>( matrix.NumMyCols() );
    }

    /*!
     * \brief Get the maximum number of entries in a row globally.
     */
    static global_ordinal_type getGlobalMaxNumRowEntries( const matrix_type& matrix )
    {
	return Teuchos::as<global_ordinal_type>( matrix.MaxNumEntries() );
    }

    /*!
     * \brief Given a local row on-process, provide the global ordinal.
     */
    static global_ordinal_type getGlobalRow( const matrix_type& matrix, 
					     const local_ordinal_type& local_row )
    { 
	MCLS_REQUIRE( matrix.RowMatrixRowMap().MyLID( local_row ) );
	return matrix.RowMatrixRowMap().GID( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static local_ordinal_type getLocalRow( const matrix_type& matrix, 
					   const global_ordinal_type& global_row )
    { 
	MCLS_REQUIRE( matrix.RowMatrixRowMap().MyGID( global_row ) );
	return matrix.RowMatrixRowMap().LID( global_row );
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static global_ordinal_type getGlobalCol( const matrix_type& matrix,
					     const local_ordinal_type& local_col )
    {
	MCLS_REQUIRE( matrix.Filled() );
	MCLS_REQUIRE( matrix.RowMatrixColMap().MyLID( local_col ) );
	return matrix.RowMatrixColMap().GID( local_col );
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static local_ordinal_type getLocalCol( const matrix_type& matrix, 
					   const global_ordinal_type& global_col )
    {
	MCLS_REQUIRE( matrix.Filled() );
	MCLS_REQUIRE( matrix.RowMatrixColMap().MyGID( global_col ) );
	return matrix.RowMatrixColMap().LID( global_col );
    }

    /*!
     * \brief Get the owning process rank for the given global rows.
     */
    static void getGlobalRowRanks( 
	const matrix_type& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_rows,
	const Teuchos::ArrayView<int>& ranks )
    {
	Teuchos::Array<local_ordinal_type> local_rows( global_rows.size() );
	int error = matrix.RowMatrixRowMap().RemoteIDList( 
            global_rows.size(),
            global_rows.getRawPtr(),
            ranks.getRawPtr(),
            local_rows.getRawPtr() );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Get the global rows owned by this proc.
     */
    static void getMyGlobalRows( 
	const matrix_type& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_rows )
    {
        int error = matrix.RowMatrixRowMap().MyGlobalElements( 
            global_rows.getRawPtr() );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Get the global columns owned by this proc.
     */
    static void getMyGlobalCols( 
	const matrix_type& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_cols )
    {
        int error = matrix.RowMatrixColMap().MyGlobalElements( 
            global_cols.getRawPtr() );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Determine whether or not a given global row is on-process.
     */
    static bool isGlobalRow( const matrix_type& matrix, 
			     const global_ordinal_type& global_row )
    {
	return matrix.RowMatrixRowMap().MyGID( global_row );
    }

    /*!
     * \brief Determine whether or not a given local row is on-process.
     */
    static bool isLocalRow( const matrix_type& matrix, 
			    const local_ordinal_type& local_row )
    { 
	return matrix.RowMatrixRowMap().MyLID( local_row );
    }

    /*!
     * \brief Determine whether or not a given global col is on-process.
     */
    static bool isGlobalCol( const matrix_type& matrix,
			     const global_ordinal_type& global_col )
    { 
	MCLS_REQUIRE( matrix.Filled() );
	return matrix.RowMatrixColMap().MyGID( global_col );
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const matrix_type& matrix, 
			    const local_ordinal_type& local_col )
    { 
	MCLS_REQUIRE( matrix.Filled() );
	return matrix.RowMatrixColMap().MyLID( local_col );
    }

    /*!
     * \brief Get a copy of a global row.
     */
    static void getGlobalRowCopy( 
	const matrix_type& matrix,
	const global_ordinal_type& global_row, 
	const Teuchos::ArrayView<global_ordinal_type>& indices,
	const Teuchos::ArrayView<scalar_type>& values,
	std::size_t& num_entries )
    {
	MCLS_REQUIRE( matrix.Filled() );
	MCLS_REQUIRE( matrix.RowMatrixRowMap().MyGID( global_row ) );
	local_ordinal_type local_row = matrix.RowMatrixRowMap().LID( global_row );
	int num_entries_int = 0;
	int error = matrix.ExtractMyRowCopy( 
            local_row, 
            Teuchos::as<local_ordinal_type>(values.size()), 
            num_entries_int,
            values.getRawPtr(), indices.getRawPtr() );
	num_entries = num_entries_int;
        MCLS_CHECK( 0 == error );

	Teuchos::ArrayView<global_ordinal_type>::iterator col_it;
	for ( col_it = indices.begin(); col_it != indices.end(); ++col_it )
	{
	    *col_it = matrix.RowMatrixColMap().GID( *col_it );
	}
    }

    /*!
     * \brief Get a copy of a local row.
     */
    static void getLocalRowCopy( 
	const matrix_type& matrix,
	const local_ordinal_type& local_row, 
	const Teuchos::ArrayView<local_ordinal_type>& indices,
	const Teuchos::ArrayView<scalar_type>& values,
	std::size_t& num_entries )
    {
	MCLS_REQUIRE( matrix.Filled() );
	MCLS_REQUIRE( matrix.RowMatrixRowMap().MyLID( local_row ) );
	int num_entries_int = 0;
	int error = matrix.ExtractMyRowCopy( 
            local_row, 
            Teuchos::as<local_ordinal_type>(values.size()), 
            num_entries_int,
            values.getRawPtr(), indices.getRawPtr() );
	num_entries = num_entries_int;
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Get a copy of the local diagonal of the matrix.
     */
    static void getLocalDiagCopy( const matrix_type& matrix, 
				  vector_type& vector )
    { 
	int error = matrix.ExtractDiagonalCopy( vector );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Matrix-Matrix multiply C = A*B
     */
    static void multiply( const Teuchos::RCP<const matrix_type>& A, 
			  const Teuchos::RCP<const matrix_type>& B, 
			  const Teuchos::RCP<matrix_type>& C,
                          bool use_transpose )
    { 
	EpetraMatrixHelpers<matrix_type>::multiply( A, B, C, use_transpose );
    }

    /*!
     * \brief Matrix-Matrix Add B = a*A + b*B.
     */
    static void add( const Teuchos::RCP<const matrix_type>& A, 
                     bool transpose_A,
                     double scalar_A,
                     const Teuchos::RCP<matrix_type>& B,
                     double scalar_B )
    { 
	EpetraMatrixHelpers<matrix_type>::add( A, transpose_A, scalar_A,
                                               B, scalar_B );
    }

    /*!
     * \brief Left-scale the matrix with a vector. A(i,j) = x(i)*A(i,j).
     */
    static void leftScale( matrix_type& A, const vector_type& x )
    { 
	int error = A.LeftScale( x );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Right-scale the matrix with a vector. A(i,j) = A(i,j)*x(j).
     */
    static void rightScale( matrix_type& A, const vector_type& x )
    { 
	int error = A.RightScale( x );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Apply the row matrix to a vector. A*x = y.
     */
    static void apply( const matrix_type& A, 
		       const vector_type& x, 
		       vector_type& y )
    {
	int error = A.Apply( x, y );
        MCLS_CHECK( 0 == error );
    }

    /*!
     * \brief Apply the transpose row matrix to a vector. (A^T)*x = y.
     */
    static void applyTranspose(  const matrix_type& A, 
                                 const vector_type& x, 
                                 vector_type& y )
    {
        bool init_state = A.UseTranspose();
        int error = const_cast<matrix_type&>(A).SetUseTranspose( true );
        MCLS_CHECK( 0 == error );
	error = A.Apply( x, y );
        MCLS_CHECK( 0 == error );
        error = const_cast<matrix_type&>(A).SetUseTranspose( init_state );
        MCLS_CHECK( 0 == error );
        MCLS_ENSURE( A.UseTranspose() == init_state );
    }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<matrix_type> copyTranspose( const matrix_type& matrix )
    { 
	return EpetraMatrixHelpers<matrix_type>::copyTranspose( matrix );
    }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const global_ordinal_type& num_neighbors )
    { 
	    return EpetraMatrixHelpers<matrix_type>::copyNearestNeighbors( 
		matrix, num_neighbors );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_EPETRAROWMATRIXADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraRowMatrixAdapter.hpp
//---------------------------------------------------------------------------//
