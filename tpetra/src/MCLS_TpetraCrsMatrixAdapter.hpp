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
#include <Teuchos_Comm.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Operator.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Import.hpp>
#include <Tpetra_RowMatrixTransposer.hpp>
#include <Tpetra_Distributor.hpp>

#include <TpetraExt_MatrixMatrix.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MatrixTraits
 * \brief Traits specialization for Tpetra::CrsMatrix.
 */
template<class Scalar, class LO, class GO>
class MatrixTraits<Tpetra::Vector<Scalar,LO,GO>, Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef typename Tpetra::CrsMatrix<Scalar,LO,GO>      matrix_type;
    typedef typename Tpetra::Vector<Scalar,LO,GO>         vector_type;
    typedef typename vector_type::scalar_type             scalar_type;
    typedef typename vector_type::local_ordinal_type      local_ordinal_type;
    typedef typename vector_type::global_ordinal_type     global_ordinal_type;
    typedef typename Tpetra::Operator<Scalar,LO,GO>       operator_type;
    typedef TpetraMatrixHelpers<Scalar,LO,GO,matrix_type> TMH;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new empty matrix from a
     * given matrix to give the new matrix the same parallel distribution as
     * the matrix parallel row distribution.
     */
    static Teuchos::RCP<matrix_type> clone( const matrix_type& matrix )
    { 
	return Tpetra::createCrsMatrix<Scalar,LO,GO>( matrix.getRowMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new matrix filled with
     * values exported from a given matrix with a parallel distribution as
     * given by the input rows.
     */
    static Teuchos::RCP<matrix_type> exportFromRows( 
        const matrix_type& matrix,
        const Teuchos::ArrayView<const global_ordinal_type>& global_rows )
    { 
        Teuchos::RCP<const Tpetra::Map<local_ordinal_type,global_ordinal_type> > 
	    target_map = 
	    Tpetra::createNonContigMap<local_ordinal_type,global_ordinal_type>( 
		global_rows(), matrix.getComm() );

        Tpetra::Import<LO,GO> importer( matrix.getRowMap(), target_map );

        return TMH::importAndFillCompleteMatrix( matrix, importer );        
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix parallel domain.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixDomain( const matrix_type& matrix )
    { 
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getDomainMap() );
    }

    /*!
     * \brief Create a reference-counted pointer to a new empty vector from a
     * matrix to give the vector the same parallel distribution as the
     * matrix range.
     */
    static Teuchos::RCP<vector_type> 
    cloneVectorFromMatrixRange( const matrix_type& matrix )
    { 
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getRangeMap() );
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
	MCLS_REQUIRE( matrix.isFillComplete() );
	return Tpetra::createVector<Scalar,LO,GO>( matrix.getColMap() );
    }

    /*!
     * \brief Get the communicator.
     */
    static Teuchos::RCP<const Teuchos::Comm<int> >
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
     * \brief Get the local number of cols.
     */
    static LO getLocalNumCols( const matrix_type& matrix )
    {
	return Teuchos::as<LO>( matrix.getColMap()->getNodeNumElements() );
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
	MCLS_REQUIRE( matrix.getRowMap()->isNodeLocalElement( local_row ) );
	return matrix.getRowMap()->getGlobalElement( local_row );
    }

    /*!
     * \brief Given a global row on-process, provide the local ordinal.
     */
    static LO getLocalRow( const matrix_type& matrix, const GO& global_row )
    { 
	MCLS_REQUIRE( matrix.getRowMap()->isNodeGlobalElement( global_row ) );
	return matrix.getRowMap()->getLocalElement( global_row );
    }

    /*!
     * \brief Given a local col on-process, provide the global ordinal.
     */
    static GO getGlobalCol( const matrix_type& matrix, const LO& local_col )
    {
	MCLS_REQUIRE( matrix.isFillComplete() );
	MCLS_REQUIRE( matrix.getColMap()->isNodeLocalElement( local_col ) );
	return matrix.getColMap()->getGlobalElement( local_col );
    }

    /*!
     * \brief Given a global col on-process, provide the local ordinal.
     */
    static LO getLocalCol( const matrix_type& matrix, const GO& global_col )
    {
	MCLS_REQUIRE( matrix.isFillComplete() );
	MCLS_REQUIRE( matrix.getColMap()->isNodeGlobalElement( global_col ) );
	return matrix.getColMap()->getLocalElement( global_col );
    }

    /*!
     * \brief Get the owning process rank for the given global rows.
     */
    static void getGlobalRowRanks( 
	const matrix_type& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_rows,
	const Teuchos::ArrayView<int>& ranks )
    {
	matrix.getRowMap()->getRemoteIndexList( global_rows, ranks );
    }

    /*!
     * \brief Get the global rows owned by this proc.
     */
    static void getMyGlobalRows( 
	const matrix_type& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_rows )
    {
        Teuchos::ArrayView<const global_ordinal_type> rows =
            matrix.getRowMap()->getNodeElementList();
        std::copy( rows.begin(), rows.end(), global_rows.begin() );
    }

    /*!
     * \brief Get the global columns owned by this proc.
     */
    static void getMyGlobalCols( 
	const matrix_type& matrix,
	const Teuchos::ArrayView<global_ordinal_type>& global_cols )
    {
        Teuchos::ArrayView<const global_ordinal_type> cols =
            matrix.getColMap()->getNodeElementList();
        std::copy( cols.begin(), cols.end(), global_cols.begin() );
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
	MCLS_REQUIRE( matrix.isFillComplete() );
	return matrix.getColMap()->isNodeGlobalElement( global_col );
    }

    /*!
     * \brief Determine whether or not a given local col is on-process.
     */
    static bool isLocalCol( const matrix_type& matrix, const LO& local_col )
    { 
	MCLS_REQUIRE( matrix.isFillComplete() );
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
	MCLS_REQUIRE( matrix.isFillComplete() );
	MCLS_REQUIRE( matrix.getRowMap()->isNodeGlobalElement( global_row ) );
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
	MCLS_REQUIRE( matrix.isFillComplete() );
	MCLS_REQUIRE( matrix.getRowMap()->isNodeLocalElement( local_row ) );
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
     * \brief Left-scale the matrix with a vector. A(i,j) = x(i)*A(i,j).
     */
    static void leftScale( matrix_type& A, const vector_type& x )
    { 
	A.leftScale( x );
    }

    /*!
     * \brief Right-scale the matrix with a vector. A(i,j) = A(i,j)*x(j).
     */
    static void rightScale( matrix_type& A, const vector_type& x )
    { 
	A.rightScale( x );
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
     * \brief Apply the transpose row matrix to a vector. (A^T)*x = y.
     */
    static void applyTranspose( const matrix_type& A, 
                                const vector_type& x, 
                                vector_type& y )
    {
        MCLS_REQUIRE( A.hasTransposeApply() );
        A.apply( x, y, Teuchos::TRANS );
    }

    /*!
     * \brief Matrix-Matrix multiply C = A*B
     */
    static void multiply( const Teuchos::RCP<const matrix_type>& A,
			  bool transpose_A,
			  const Teuchos::RCP<const matrix_type>& B,
			  bool transpose_B,
			  const Teuchos::RCP<matrix_type>& C )
    {
	Tpetra::MatrixMatrix::Multiply( *A, transpose_A, *B, transpose_B, *C );
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
        // Currently not implemented.
    }

    /*!
     * \brief Get a copy of the transpose of a matrix.
     */
    static Teuchos::RCP<matrix_type> copyTranspose( const matrix_type& matrix )
    { 
	Teuchos::RCP<const matrix_type> matrix_rcp = 
	    Teuchos::rcpFromRef( matrix );
	Tpetra::RowMatrixTransposer<Scalar,LO,GO> transposer( matrix_rcp );
	return transposer.createTranspose();
    }

    /*
     * \brief Create a reference-counted pointer to a new matrix with a
     * specified number of off-process nearest-neighbor global rows.
     */
    static Teuchos::RCP<matrix_type> copyNearestNeighbors( 
    	const matrix_type& matrix, const GO& num_neighbors )
    { 
	MCLS_REQUIRE( num_neighbors >= 0 ); 

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

	MCLS_ENSURE( !neighbor_matrix.is_null() );
	MCLS_ENSURE( neighbor_matrix->isFillComplete() );
	return neighbor_matrix;
    }

};

//---------------------------------------------------------------------------//

#endif // end MCLS_TPETRACRSMATRIXADAPTER_HPP

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_TpetraCrsMatrixAdapter.hpp
//---------------------------------------------------------------------------//
