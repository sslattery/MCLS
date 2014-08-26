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
 * \file MCLS_EpetraPointJacobiPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief Point Jacobi preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include "MCLS_EpetraPointJacobiPreconditioner.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Epetra_Vector.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the valid parameters for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraPointJacobiPreconditioner::getValidParameters() const
{
    return Teuchos::parameterList();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraPointJacobiPreconditioner::getCurrentParameters() const
{
    // This preconditioner has no parameters.
    return Teuchos::parameterList();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraPointJacobiPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    // This preconditioner has no parameters.
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraPointJacobiPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Build the preconditioner.
 */
void EpetraPointJacobiPreconditioner::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    // Create the preconditioner.
    d_preconditioner = Teuchos::rcp( 
	new Epetra_CrsMatrix(Copy,d_A->RowMatrixRowMap(),1,true) );
	    
    // Compute the inverse of the diagonal.
    Teuchos::RCP<Epetra_Vector> diagonal = 
	Teuchos::rcp( new Epetra_Vector( d_A->RowMatrixRowMap() ) );

    MCLS_CHECK_ERROR_CODE(
	d_A->ExtractDiagonalCopy( *diagonal )
	);
    MCLS_CHECK_ERROR_CODE(
	diagonal->Reciprocal( *diagonal )
	);

    // Build a matrix from the diagonal vector.
    Teuchos::Array<int> rows( d_preconditioner->RowMap().NumMyElements() );
    d_preconditioner->RowMap().MyGlobalElements( rows.getRawPtr() );
    Teuchos::Array<int>::const_iterator row_it;
    Teuchos::Array<int> col(1);
    int local_row = 0;
    for ( row_it = rows.begin(); row_it != rows.end(); ++row_it )
    {
	col[0] = *row_it;
	MCLS_CHECK_ERROR_CODE(
	    d_preconditioner->InsertGlobalValues( 
		*row_it, 1, &(*diagonal)[local_row], col.getRawPtr() )
	    );
	++local_row;
    }

    MCLS_CHECK_ERROR_CODE(
	d_preconditioner->FillComplete() 
	);
	
    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
    MCLS_ENSURE( d_preconditioner->Filled() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraPointJacobiPreconditioner.cpp
//---------------------------------------------------------------------------//
