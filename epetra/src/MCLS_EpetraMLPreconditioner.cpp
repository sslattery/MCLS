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

#include <algorithm>

#include "MCLS_EpetraMLPreconditioner.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>

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
    std::cout << "MCLS ML: Building ML Preconditioner" << std::endl;
    Teuchos::Time timer("");
    timer.start(true);

    ML_Epetra::MultiLevelPreconditioner ml( *d_A, 
                                            d_plist->sublist("ML Settings") );

    // Extract the preconditioner.
    Teuchos::RCP<Epetra_CrsMatrix> ml_extract = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, d_A->RowMatrixRowMap(), 0) );

    int num_rows = d_A->NumMyRows();
    Epetra_Vector basis( d_A->RowMatrixRowMap() );
    Epetra_Vector extract_row( d_A->RowMatrixRowMap() );
    Teuchos::ArrayRCP<double> values( num_rows );
    Teuchos::ArrayRCP<double>::iterator value_iterator;
    Teuchos::ArrayRCP<int> indices( num_rows );
    Teuchos::ArrayRCP<int>::iterator index_iterator;
    int values_size = 0;
    int indices_size = 0;
    int col_counter = 0;

    // Invert the matrix row-by-row.
    std::cout << "MCLS ML: Extracting ML Preconditioner" << std::endl;
    int error = 0;
    for ( int i = 0; i < num_rows; ++i )
    {
        // Set the basis for this row.
        basis.PutScalar(0.0);
        basis[i] = 1.0;
            
        // Get the row for the preconditioner.
        error = ml.ApplyInverse( basis, extract_row );
        MCLS_CHECK( 0 == error );

        // Get a view of the extracted row.
        error = extract_row.ExtractCopy( values.getRawPtr() );
        MCLS_CHECK( 0 == error );

        // Filter any zero values
        col_counter = 0;
        for ( value_iterator = values.begin(),
              index_iterator = indices.begin();
              value_iterator != values.end();
              ++value_iterator, ++index_iterator )
        {
            if ( 0.0 == *value_iterator )
            {
                *index_iterator = -1;
            }
            else
            {
                *index_iterator = d_A->RowMatrixColMap().GID( col_counter );
            }
            ++col_counter;
        }
        value_iterator = 
            std::remove( values.begin(), values.end(), 0.0 );
        index_iterator = 
            std::remove( indices.begin(), indices.end(), -1 );

        // Check for degeneracy and consistency.
        values_size = std::distance( values.begin(), value_iterator );
        MCLS_REMEMBER( indices_size = std::distance(indices.begin(), index_iterator) );
        MCLS_CHECK( values_size > 0 );
        MCLS_CHECK( values_size == indices_size );

        // Populate the row in the preconditioner matrix.
        error = ml_extract->InsertGlobalValues( d_A->RowMatrixRowMap().GID(i),
                                                values_size,
                                                values.getRawPtr(),
                                                indices.getRawPtr() );
        MCLS_CHECK( 0 == error );
    }
    error = ml_extract->FillComplete();
    MCLS_CHECK( 0 == error );

    // Cleanup.
    values.clear();
    indices.clear();

    // Compute the tranpose as we really extracted columns above.
    std::cout << "MCLS ML: Transposing ML Preconditioner" << std::endl;
    Epetra_RowMatrixTransposer transposer( ml_extract.getRawPtr() );
    Epetra_CrsMatrix* transpose_matrix;
    error = transposer.CreateTranspose( true, transpose_matrix );
    MCLS_CHECK( 0 == error );
    MCLS_ENSURE( transpose_matrix->Filled() );
    d_preconditioner = Teuchos::RCP<Epetra_CrsMatrix>( transpose_matrix );
    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );

    timer.stop();
    std::cout << "MCLS ML: Complete in " << timer.totalElapsedTime() 
              << " seconds." << std::endl;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraMLPreconditioner.cpp
//---------------------------------------------------------------------------//
