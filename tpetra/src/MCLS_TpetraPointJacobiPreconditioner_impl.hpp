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
 * \file MCLS_TpetraPointJacobiPreconditioner_impl.hpp
 * \author Stuart R. Slattery
 * \brief Point Jacobi preconditioning for Tpetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAPOINTJACOBI_IMPL_HPP
#define MCLS_TPETRAPOINTJACOBI_IMPL_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the valid parameters for this preconditioner.
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<const Teuchos::ParameterList> 
TpetraPointJacobiPreconditioner<Scalar,LO,GO>::getValidParameters() const
{
    return Teuchos::parameterList();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<const Teuchos::ParameterList> 
TpetraPointJacobiPreconditioner<Scalar,LO,GO>::getCurrentParameters() const
{
    // This preconditioner has no parameters.
    return Teuchos::parameterList();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
template<class Scalar, class LO, class GO>
void TpetraPointJacobiPreconditioner<Scalar,LO,GO>::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    // This preconditioner has no parameters.
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
template<class Scalar, class LO, class GO>
void TpetraPointJacobiPreconditioner<Scalar,LO,GO>::setOperator( 
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
void TpetraPointJacobiPreconditioner<Scalar,LO,GO>::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->isFillComplete() );

    // Create the preconditioner.
    d_preconditioner = 
	Tpetra::createCrsMatrix<Scalar,LO,GO>( d_A->getRowMap(), 1 );

    // Compute the inverse of the diagonal.
    Teuchos::RCP<Tpetra::Vector<Scalar,LO,GO> > diagonal = 
	Tpetra::createVector<Scalar,LO,GO>( d_A->getRowMap() );
    d_A->getLocalDiagCopy( *diagonal );
    diagonal->reciprocal( *diagonal );
    Teuchos::ArrayRCP<const Scalar> diagonal_data = diagonal->getData();

    // Build a matrix from the diagonal vector.
    Teuchos::ArrayView<const GO> rows = 
	d_preconditioner->getRowMap()->getNodeElementList();
    typename Teuchos::ArrayView<const GO>::const_iterator row_it;
    Teuchos::Array<GO> col(1);
    LO local_row = 0;
    for ( row_it = rows.begin(); row_it != rows.end(); ++row_it )
    {
	col[0] = *row_it;
	d_preconditioner->insertGlobalValues( 
	    *row_it, col(), diagonal_data( local_row, 1 ) );
	++local_row;
    }

    d_preconditioner->fillComplete();
	
    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
    MCLS_ENSURE( d_preconditioner->isFillComplete() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAPOINTJACOBI_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraPointJacobiPreconditioner_impl.hpp
//---------------------------------------------------------------------------//
