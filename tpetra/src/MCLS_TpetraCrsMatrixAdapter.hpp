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
 * \file MCLS_TpetraCrsMatrixAdapter.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra::CrsMatrix Adapter.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRACRSMATRIXADAPTER_HPP
#define MCLS_TPETRACRSMATRIXADAPTER_HPP

#include <algorithm>

#include <MCLS_DBC.hpp>
#include <MCLS_MatrixTraits.hpp>
#include <MCLS_TpetraHelpers.hpp>

#include <Teuchos_as.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_RowMatrixTransposer.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class VectorTraits
 * \brief Traits specialization for Tpetra::Vector.
 */
template<class Scalar, class LO, class GO>
class MatrixTraits<Scalar,LO,GO,Tpetra::Vector<Scalar,LO,GO>,
		   Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef typename Tpetra::CrsMatrix<Scalar,LO,GO>      matrix_type;
    typedef typename Tpetra::Vector<Scalar,LO,GO>         vector_type;
    typedef typename vector_type::scalar_type             scalar_type;
    typedef typename vector_type::local_ordinal_type      local_ordinal_type;
    typedef typename vector_type::global_ordinal_type     global_ordinal_type;
    typedef TpetraMatrixHelpers<Scalar,LO,GO,matrix_type> TMH;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty matrix with
     * the same parallel distribution as the given matrix.
     */
    static Teuchos::RCP<matrix_type> clone( const matrix_type& matrix )
    { 
	return Tpetra::createCrsMatrix<Scalar,LO,GO>( matrix.getMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel row distribution.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixRows( const matrix_type& matrix )
    { 
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getRowMap() );
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
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getColMap() );
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
     * \brief Get a view of a global row.
     */
    static void getGlobalRowView( const matrix_type& matrix,
				  const GO& global_row, 
				  Teuchos::ArrayView<const GO> &indices,
				  Teuchos::ArrayView<const Scalar> &values )
    {
	Require( !matrix.isFillComplete() );
	Require( matrix.getRowMap()->isNodeGlobalElement( global_row ) );
	matrix.getGlobalRowView( global_row, indices, values );
    }

    /*!
     * \brief Get a view of a local row.
     */
    static void getLocalRowView( const matrix_type& matrix,
				 const LO& local_row, 
				 Teuchos::ArrayView<const LO> &indices,
				 Teuchos::ArrayView<const Scalar> &values )
    {
	Require( matrix.isFillComplete() );
	Require( matrix.getRowMap()->isNodeLocalElement( local_row ) );
	matrix.getLocalRowView( local_row, indices, values );
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
	Tpetra::RowMatrixTransposer<Scalar,LO,GO> transposer( matrix );
	return transposer.createTranspose();
    }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const GO& num_neighbors )
    { 
	Require( num_neighbors >= 0 ); 

	// Setup for neighbor construction.
	Teuchos::RCP<const Tpetra::Map<LO,GO> > empty_map = 
	    Tpetra::createUniformContigMap<LO,GO>( 
		0, matrix.getComm() );
	Teuchos::RCP<matrix_type> neighbor_matrix = 
	    Tpetra::createCrsMatrix<Scalar,LO,GO>( empty_map );
	neighbor_matrix->fillComplete();

	Teuchos::ArrayView<const GO> global_rows;
	typename Teuchos::ArrayView<const GO>::const_iterator global_rows_it;
	typename Teuchos::Array<GO>::iterator ghost_global_bound;

	// Get the initial off proc columns.
	Teuchos::Array<GO> ghost_global_rows =
	    TMH::getOffProcColsAsRows( matrix );

	// Build the neighbors by traversing the graph.
	for ( GO i = 0; i < num_neighbors; ++i )
	{
	    // Get rid of the global rows that belong to the original
	    // matrix. We don't need to store these, just the neighbors.
	    global_rows = matrix.getRowMap()->getNodeElementList();
	    for ( global_rows_it = global_rows.begin();
		  global_rows_it != global_rows.end();
		  ++global_rows_it )
	    {
		ghost_global_bound = std::remove( ghost_global_rows.begin(), 
						  ghost_global_rows.end(), 
						  *global_rows_it );
		ghost_global_rows.resize( std::distance(ghost_global_rows.begin(),
							ghost_global_bound) );
	    }

	    // Get the current set of global rows in the neighbor matrix. 
	    global_rows = neighbor_matrix->getRowMap()->getNodeElementList();

	    // Append the on proc neighbor columns to the off proc columns.
	    for ( global_rows_it = global_rows.begin();
		  global_rows_it != global_rows.end();
		  ++global_rows_it )
	    {
		ghost_global_rows.push_back( *global_rows_it );
	    }
	
	    // Make a new map of the combined global rows and off proc columns.
	    Teuchos::RCP<const Tpetra::Map<LO,GO> > ghost_map = 
		Tpetra::createNonContigMap<LO,GO>( 
		    ghost_global_rows(), neighbor_matrix->getComm() );

	    // Import the neighbor matrix with the new neighbor.
	    Tpetra::Import<LO,GO> ghost_importer( 
		matrix.getRowMap(), ghost_map );

	    neighbor_matrix = 
		TMH::importAndFillCompleteMatrix( matrix, ghost_importer );

	    // Get the next rows in the graph.
	    ghost_global_rows = TMH::getOffProcColsAsRows( *neighbor_matrix );
	}

	Ensure( !neighbor_matrix.is_null() );
	Ensure( neighbor_matrix.isFillComplete() );
	return neighbor_matrix;
    }
};

//---------------------------------------------------------------------------//

#endif // end MCLS_TPETRACRSMATRIXADAPTER_HPP

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_TpetraCrsMatrixAdapter.hpp
//---------------------------------------------------------------------------//
