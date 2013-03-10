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

#include <Teuchos_Array.hpp>
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
    plist->set<int>("Jacobi Block Size", 0);
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

    // Build the block preconditioner.
    d_preconditioner = Teuchos::rcp( 
	new Epetra_CrsMatrix( Copy, d_A->RowMatrixRowMap(), block_size ) );

    // Populate the preconditioner with inverted blocks.
    Teuchos::SerialDenseMatrix<int,double> block( block_size, block_size );
    int col_start = 0;
    int global_row = 0;
    Teuchos::Array<int> block_cols( block_size );
    for ( int n = 0; n < num_blocks; ++n )
    {
	// Starting row/column for the block.
	col_start = block_size*n;

	// Extract the block. Note that I form the tranposed local block to
	// facilitate constructing the preconditioner in the second group of
	// loops. I grab each individual element here because I want the
	// zero's to build the block, but there's probably a cheaper way of
	// doing the extraction.
	for ( int i = 0; i < block_size; ++i )
	{
	    global_row = d_A->RowMatrixRowMap().GID(col_start+i);
	    for ( int j = 0; j < block_size; ++j )
	    {
		block_cols[j] = d_A->RowMatrixColMap().GID(col_start+j);
	    }
	    for ( int j = 0; j < block_size; ++j )
	    {
		block(j,i) = 
		    getMatrixComponentFromGlobal( d_A, global_row, block_cols[j] );
	    }
	}

	// Invert the block.
	invertSerialDenseMatrix( block );

	// Add the block to the preconditioner.
	for ( int i = 0; i < block_size; ++i )
	{
	    global_row = 
		d_preconditioner->RowMatrixRowMap().GID(col_start+i);

	    d_preconditioner->InsertGlobalValues( 
		global_row, block_size, block[i], block_cols.getRawPtr() );
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
    int ipiv = 0;
    int info = 0;
    lapack.GETRF( block.numRows(), block.numCols(), block.values(), 
		  block.stride(), &ipiv, &info );
    MCLS_CHECK( info == 0 );

    // Compute the inverse of the block from the LU-factorization.
    Teuchos::Array<double> work( block.numRows() );
    lapack.GETRI( 
        block.numCols(), block.values(), block.stride(), 
	&ipiv, work.getRawPtr(), work.size(), &info );
    MCLS_CHECK( info == 0 );
    MCLS_CHECK( work[0] == block.numRows() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a local component of an operator given a global row and column
 * index. 
 */
double EpetraBlockJacobiPreconditioner::getMatrixComponentFromGlobal( 
    const Teuchos::RCP<const matrix_type>& matrix,
    const int global_row, const int global_col )
{
    const Epetra_Map row_map = matrix->RowMatrixRowMap();
    const Epetra_Map col_map = matrix->RowMatrixColMap();

    MCLS_REQUIRE( row_map.MyGID(global_row) );

    int local_row = row_map.LID( global_row );
    int local_col = col_map.LID( global_col );

    // If the block column is not local, then we get a zero for this entry.
    if ( local_col == Teuchos::OrdinalTraits<int>::invalid() )
    {
	return 0.0;
    }

    int max_size = matrix->MaxNumEntries();
    int num_entries = 0;
    Teuchos::Array<int> local_indices( max_size );
    Teuchos::Array<double> local_values( max_size );
    matrix->ExtractMyRowCopy( local_row, max_size, num_entries,
			      local_values.getRawPtr(), 
			      local_indices.getRawPtr() );
    local_values.resize( num_entries );
    local_indices.resize( num_entries );

    Teuchos::Array<int>::iterator local_idx_it =
	std::find( local_indices.begin(), local_indices.end(), local_col );

    if ( local_idx_it != local_indices.end() )
    {
	return local_values[ std::distance( local_indices.begin(),
					    local_idx_it ) ];
    }

    return 0.0;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraBlockJacobiPreconditioner.cpp
//---------------------------------------------------------------------------//
