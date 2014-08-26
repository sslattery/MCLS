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
 * \file MCLS_EpetraBlockJacobiPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief Block Jacobi preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include <algorithm>

#include "MCLS_EpetraBlockJacobiPreconditioner.hpp"

#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_LAPACK.hpp>
#include <Teuchos_OrdinalTraits.hpp>

#include <Epetra_Map.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
EpetraBlockJacobiPreconditioner::EpetraBlockJacobiPreconditioner(
    const Teuchos::RCP<Teuchos::ParameterList>& params )
    : d_plist( params )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the valid parameters for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraBlockJacobiPreconditioner::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size", 1);
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraBlockJacobiPreconditioner::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraBlockJacobiPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraBlockJacobiPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the preconditioner.
 */
void EpetraBlockJacobiPreconditioner::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    // Get the block size.
    int block_size = d_plist->get<int>("Jacobi Block Size");

    // We require that all blocks are local.
    MCLS_REQUIRE( d_A->NumMyRows() % block_size == 0 );

    // Get the number of blocks.
    int num_blocks = d_A->NumMyRows() / block_size;

    // Get the global rows on this proc. We'll sort them for building the
    // blocks as the blocks should be contiguous in global row indexing.
    Teuchos::Array<int> global_rows( d_A->NumMyRows() );
    d_A->RowMatrixRowMap().MyGlobalElements( global_rows.getRawPtr() );
    std::sort( global_rows.begin(), global_rows.end() );

    // Build the block preconditioner.
    d_preconditioner = Teuchos::rcp( 
	new Epetra_CrsMatrix( Copy, d_A->RowMatrixRowMap(), block_size ) );

    // Populate the preconditioner with inverted blocks.
    Teuchos::SerialDenseMatrix<int,double> block( block_size, block_size );
    int col_start = 0;
    Teuchos::Array<int> block_cols( block_size );
    for ( int n = 0; n < num_blocks; ++n )
    {
	// Starting row/column for the block.
	col_start = block_size*n;

	// Extract the block. Note that I form the tranposed local block to
	// facilitate constructing the preconditioner in the second group of
	// loops. 
	for ( int i = 0; i < block_size; ++i )
	{
	    for ( int j = 0; j < block_size; ++j )
	    {
		block_cols[j] = global_rows[col_start]+j;
	    }

            getBlockRowFromGlobal( 
                block, i, d_A, global_rows[col_start+i], block_cols );
	}

	// Invert the block.
	invertSerialDenseMatrix( block );

	// Add the block to the preconditioner.
	for ( int i = 0; i < block_size; ++i )
	{
	    MCLS_CHECK_ERROR_CODE(
		d_preconditioner->InsertGlobalValues( 
		    global_rows[col_start+i], block_size, 
		    block[i], block_cols.getRawPtr() )
		);
	}
    }

    d_preconditioner->FillComplete();

    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
    MCLS_ENSURE( d_preconditioner->Filled() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Invert a Teuchos::SerialDenseMatrix block.
 */
void EpetraBlockJacobiPreconditioner::invertSerialDenseMatrix(
    Teuchos::SerialDenseMatrix<int,double>& block )
{
    // Make a LAPACK object.
    Teuchos::LAPACK<int,double> lapack;

    // Compute the LU-factorization of the block.
    Teuchos::ArrayRCP<int> ipiv( block.numRows() );
    int info = 0;
    lapack.GETRF( block.numRows(), block.numCols(), block.values(), 
		  block.stride(), ipiv.getRawPtr(), &info );
    MCLS_CHECK( info == 0 );

    // Compute the inverse of the block from the LU-factorization.
    Teuchos::ArrayRCP<double> work( block.numRows() );
    lapack.GETRI( 
        block.numCols(), block.values(), block.stride(), 
	ipiv.getRawPtr(), work.getRawPtr(), work.size(), &info );
    MCLS_CHECK( info == 0 );
    MCLS_CHECK( work[0] == block.numRows() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a row of an operator in a block given a global row and block
 * column indices.
 */
void EpetraBlockJacobiPreconditioner::getBlockRowFromGlobal( 
    Teuchos::SerialDenseMatrix<int,double>& block,
    const int block_row,
    const Teuchos::RCP<const matrix_type>& matrix,
    const int global_row,
    const Teuchos::Array<int>& global_cols )
{
    const Epetra_Map row_map = matrix->RowMatrixRowMap();
    const Epetra_Map col_map = matrix->RowMatrixColMap();

    MCLS_REQUIRE( row_map.MyGID(global_row) );

    // Extract the local row.
    int local_row = row_map.LID( global_row );
    int max_size = matrix->MaxNumEntries();
    int num_entries = 0;
    Teuchos::Array<int> local_indices( max_size );
    Teuchos::Array<double> local_values( max_size );
    MCLS_CHECK_ERROR_CODE(
	matrix->ExtractMyRowCopy( local_row, max_size, num_entries,
				  local_values.getRawPtr(), 
				  local_indices.getRawPtr() )
	);
    local_values.resize( num_entries );
    local_indices.resize( num_entries );

    // Load the row into the block.
    int block_col = 0;
    Teuchos::Array<int>::const_iterator global_col_it;
    for ( global_col_it = global_cols.begin();
          global_col_it != global_cols.end();
          ++global_col_it )
    {
        MCLS_CHECK( block_col < block.numRows() );

        // Get the local column.
        int local_col = col_map.LID( *global_col_it );

        // See if there's a non-zero entry for this column.
        Teuchos::Array<int>::iterator local_idx_it =
            std::find( local_indices.begin(), local_indices.end(), local_col );

        if ( local_idx_it != local_indices.end() )
        {
            block( block_col, block_row ) = local_values[ 
                std::distance( local_indices.begin(), local_idx_it ) ];
        }

        ++block_col;
    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraBlockJacobiPreconditioner.cpp
//---------------------------------------------------------------------------//
