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
 * \file MCLS_EpetraILUTPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief ILUT preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include "MCLS_EpetraILUTPreconditioner.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>

#include <Epetra_Vector.h>

#include <Ifpack_ILUT.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
EpetraILUTPreconditioner::EpetraILUTPreconditioner(
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
EpetraILUTPreconditioner::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<double>("fact: ilut level-of-fill", 1.0);
    plist->set<double>("fact: drop tolerance", 1.0e-2);
    plist->set<double>("fact: absolute threshold", 1.0 );
    plist->set<double>("fact: relative threshold", 1.0 );
    plist->set<double>("fact: relax value", 1.0 );
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraILUTPreconditioner::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraILUTPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraILUTPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Build the preconditioner.
 */
void EpetraILUTPreconditioner::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    // Build the Ifpack ILUT preconditioner.
    Ifpack_ILUT ifpack( d_A.getRawPtr() );
    ifpack.SetParameters( *d_plist );
    ifpack.Initialize();
    ifpack.Compute();

    // Invert L and U.
    d_l_inv = computeTriInverse( ifpack.L(), false );
    d_u_inv = computeTriInverse( ifpack.U(), true );

    MCLS_ENSURE( Teuchos::nonnull(d_l_inv) );
    MCLS_ENSURE( Teuchos::nonnull(d_u_inv) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the inverse of a tri-diagonal matrix from Ifpack.
 */
Teuchos::RCP<Epetra_CrsMatrix>
EpetraILUTPreconditioner::computeTriInverse( const Epetra_CrsMatrix& A,
                                             bool is_upper )
{
    Teuchos::RCP<Epetra_CrsMatrix> inverse = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, A.RowMatrixRowMap(), 0) );

    int num_rows = A.NumMyRows();
    Epetra_Vector basis( A.RowMatrixRowMap() );
    Epetra_Vector inverse_row( A.RowMatrixRowMap() );
    Teuchos::Array<double> values;
    Teuchos::Array<int> indices;

    // Invert the matrix row-by-row.
    for ( int i = 0; i < num_rows; ++i )
    {
        // Set the basis for this row.
        basis.PutScalar(0.0);
        basis[i] = 1.0;
            
        // Get the row for the inverse.
        A.Solve( is_upper, true, false, basis, inverse_row );

        // Get the non-zero elements of the row.
        for ( int j = 0; j < num_rows; ++j )
        {
            if ( inverse_row[j] != 0.0 )
            {
                values.push_back( inverse_row[j] );
                indices.push_back( A.RowMatrixRowMap().GID(j) );
            }           
        }

        // Populate the row in the inverse matrix.
        inverse->InsertGlobalValues( A.RowMatrixRowMap().GID(i),
                                     values.size(),
                                     values.getRawPtr(),
                                     indices.getRawPtr() );

        values.clear();
        indices.clear();
    }

    inverse->FillComplete();

    return inverse;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraILUTPreconditioner.cpp
//---------------------------------------------------------------------------//
