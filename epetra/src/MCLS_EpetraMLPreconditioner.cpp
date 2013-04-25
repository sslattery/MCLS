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
 * \file MCLS_EpetraMLPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief ML preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include "MCLS_EpetraMLPreconditioner.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>

#include <Epetra_Vector.h>
#include <Epetra_RowMatrixTransposer.h>

#include <ml_MultiLevelPreconditioner.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
EpetraMLPreconditioner::EpetraMLPreconditioner(
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
EpetraMLPreconditioner::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraMLPreconditioner::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraMLPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraMLPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Build the preconditioner.
 */
void EpetraMLPreconditioner::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    // Build the ML preconditioner.
    ML_Epetra::MultiLevelPreconditioner ml( *d_A, *d_plist );

    // Extract the preconditioner.
    Teuchos::RCP<Epetra_CrsMatrix> ml_extract = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, d_A->RowMatrixRowMap(), 0) );

    int num_rows = d_A->NumMyRows();
    Epetra_Vector basis( d_A->RowMatrixRowMap() );
    Epetra_Vector extract_row( d_A->RowMatrixRowMap() );
    Teuchos::Array<double> values;
    Teuchos::Array<int> indices;

    // Invert the matrix row-by-row.
    int error = 0;
    for ( int i = 0; i < num_rows; ++i )
    {
        // Set the basis for this row.
        basis.PutScalar(0.0);
        basis[i] = 1.0;
            
        // Get the row for the preconditioner.
        error = ml.ApplyInverse( basis, extract_row );
        MCLS_CHECK( 0 == error );

        // Get the non-zero elements of the row.
        for ( int j = 0; j < num_rows; ++j )
        {
            if ( extract_row[j] != 0.0 )
            {
                values.push_back( extract_row[j] );
                indices.push_back( d_A->RowMatrixRowMap().GID(j) );
            }           
        }

        // Populate the row in the ml_extract matrix.
        error = ml_extract->InsertGlobalValues( 
            d_A->RowMatrixRowMap().GID(i),
            values.size(),
            values.getRawPtr(),
            indices.getRawPtr() );
        MCLS_CHECK( 0 == error );

        values.clear();
        indices.clear();
    }
    error = ml_extract->FillComplete();
    MCLS_CHECK( 0 == error );

    // Compute the tranpose as we really extracted columns above.
    Epetra_RowMatrixTransposer transposer( ml_extract.getRawPtr() );
    Epetra_CrsMatrix* transpose_matrix;
    error = transposer.CreateTranspose( true, transpose_matrix );
    MCLS_CHECK( 0 == error );
    MCLS_ENSURE( transpose_matrix->Filled() );
    d_preconditioner = Teuchos::RCP<Epetra_CrsMatrix>( transpose_matrix );
    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraMLPreconditioner.cpp
//---------------------------------------------------------------------------//
