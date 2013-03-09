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
 * \file MCLS_TpetraBlockJacobiPreconditioner_impl.hpp
 * \author Stuart R. Slattery
 * \brief Block Jacobi preconditioning for Tpetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRABLOCKJACOBI_IMPL_HPP
#define MCLS_TPETRABLOCKJACOBI_IMPL_HPP

#include <algorithm>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_LAPACK.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Scalar, class LO, class GO>
TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::TpetraBlockJacobiPreconditioner(
    const Teuchos::RCP<Teuchos::ParameterList>& params )
    : d_plist( params )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the valid parameters for this preconditioner.
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<const Teuchos::ParameterList> 
TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size",0);
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<const Teuchos::ParameterList> 
TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
template<class Scalar, class LO, class GO>
void TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
template<class Scalar, class LO, class GO>
void TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the preconditioner.
 */
template<class Scalar, class LO, class GO>
void TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->isFillComplete() );

    // Get the block size.
    int block_size = d_plist->get<int>("Jacobi Block Size");

    // We require that all blocks are local.
    MCLS_REQUIRE( d_A->getRowMap()->getNodeNumElements() % block_size == 0 );

    // Get the number of blocks.
    int num_blocks = d_A->getRowMap()->getNodeNumElements() / block_size;

    // Build the block preconditioner.
    d_preconditioner = 
	Tpetra::createCrsMatrix<Scalar,LO,GO>( d_A->getRowMap() );

    // Populate the preconditioner with inverted blocks.
    Teuchos::SerialDenseMatrix<int,Scalar> block( block_size, block_size );
    GO col_start = 0;
    GO global_row = 0;
    Teuchos::Array<GO> block_cols( block_size );
    for ( int n = 0; n < num_blocks; ++n )
    {
	// Starting row/column for the block.
	col_start = block_size*n;

	// Extract the block. Note that I form the tranposed block to
	// facilitate constructing the preconditioner in the second group of
	// loops. I grab each individual element here because I want the
	// zero's to build the block, but there's probably a cheaper way of
	// doing the extraction.
	for ( int i = 0; i < block_size; ++i )
	{
	    global_row = d_A->getRowMap()->getGlobalElement(col_start+i);
	    for ( int j = 0; j < block_size; ++j )
	    {
		block_cols[j] = d_A->getColMap()->getGlobalElement(col_start+j);
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
		d_preconditioner->getRowMap()->getGlobalElement(col_start+i);

	    d_preconditioner->insertGlobalValues( 
		global_row, block_cols(), 
		Teuchos::ArrayView<Scalar>(block[i], block_size) );
	}
    }

    d_preconditioner->fillComplete();

    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
    MCLS_ENSURE( d_preconditioner->isFillComplete() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Invert a Teuchos::SerialDenseMatrix block.
 */
template<class Scalar, class LO, class GO>
void TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::invertSerialDenseMatrix(
    Teuchos::SerialDenseMatrix<int,Scalar>& block )
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
 * \brief Get a local component of an operator given a local row and column
 * index. 
 */
template<class Scalar, class LO, class GO>
Scalar TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::getMatrixComponentFromLocal( 
    const Teuchos::RCP<const matrix_type>& matrix, 
    const LO local_row, const LO local_col )
{
    MCLS_REQUIRE( matrix->getRowMap()->isNodeLocalElement( local_row ) );
    MCLS_REQUIRE( matrix->getColMap()->isNodeLocalElement( local_col ) );

    Teuchos::ArrayView<const LO> local_indices;
    Teuchos::ArrayView<const Scalar> local_values;
    matrix->getLocalRowView( local_row, local_indices, local_values );

    typename Teuchos::ArrayView<const LO>::const_iterator local_idx_it =
	std::find( local_indices.begin(), local_indices.end(), local_col );

    if ( local_idx_it != local_indices.end() )
    {
	return local_values[ std::distance( local_indices.begin(),
					    local_idx_it ) ];
    }
    else
    {
	return 0.0;
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a local component of an operator given a global row and column
 * index. 
 */
template<class Scalar, class LO, class GO>
Scalar TpetraBlockJacobiPreconditioner<Scalar,LO,GO>::getMatrixComponentFromGlobal( 
    const Teuchos::RCP<const matrix_type>& matrix,
    const GO global_row, const GO global_col )
{
    Teuchos::RCP<const Tpetra::Map<LO,GO> > row_map = matrix->getRowMap();
    Teuchos::RCP<const Tpetra::Map<LO,GO> > col_map = matrix->getColMap();

    LO local_row = row_map->getLocalElement( global_row );
    LO local_col = col_map->getLocalElement( global_col );

    MCLS_REQUIRE( local_row != Teuchos::OrdinalTraits<LO>::invalid() );
    MCLS_REQUIRE( local_col != Teuchos::OrdinalTraits<LO>::invalid() );

    Teuchos::ArrayView<const LO> local_indices;
    Teuchos::ArrayView<const Scalar> local_values;
    matrix->getLocalRowView( local_row, local_indices, local_values );

    typename Teuchos::ArrayView<const LO>::const_iterator local_idx_it =
	std::find( local_indices.begin(), local_indices.end(), local_col );

    if ( local_idx_it != local_indices.end() )
    {
	return local_values[ std::distance( local_indices.begin(),
					    local_idx_it ) ];
    }
    else
    {
	return 0.0;
    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRABLOCKJACOBI_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraBlockJacobiPreconditioner_impl.hpp
//---------------------------------------------------------------------------//
