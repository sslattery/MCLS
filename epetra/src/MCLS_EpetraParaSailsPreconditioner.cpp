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
 * \file MCLS_EpetraParaSailsPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief ParaSails preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include "MCLS_EpetraParaSailsPreconditioner.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Epetra_Vector.h>

#include <ParaSails.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#endif

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
EpetraParaSailsPreconditioner::EpetraParaSailsPreconditioner(
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
EpetraParaSailsPreconditioner::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    d_plist->set<double>("ParaSails: Threshold", 0.0);
    d_plist->set<int>("ParaSails: Number of Levels", 0.0);
    d_plist->set<double>("ParaSails: Filter", 0.0);
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraParaSailsPreconditioner::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraParaSailsPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraParaSailsPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Build the preconditioner.
 */
void EpetraParaSailsPreconditioner::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    // Get the ParaSails parameters.
    double threshold = d_plist->get<double>("ParaSails: Threshold");
    int num_levels = d_plist->get<int>("ParaSails: Number of Levels");
    double filter = d_plist->get<double>("ParaSails: Filter");

    // Extract the raw MPI handle.
    Teuchos::RCP<const Epetra_Comm> epetra_comm = 
        Teuchos::rcp( &(d_A->Comm()), false );
    Teuchos::RCP<const Epetra_MpiComm> mpi_epetra_comm =
        Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>( epetra_comm );
    MPI_Comm raw_mpi_comm = mpi_epetra_comm->Comm();

    // Create a ParaSails matrix from the operator. Right now we'll assume the
    // global indices are contiguous.
    Teuchos::ArrayRCP<double> values( d_A->MaxNumEntries() );
    Teuchos::ArrayRCP<int> indices( d_A->MaxNumEntries() );
    int error = 0;
    int num_entries = 0;
    int local_row = 0;
    int beg_row = d_A->RowMatrixRowMap().MinMyGID();
    int end_row = d_A->RowMatrixRowMap().MaxMyGID();
    Teuchos::ArrayRCP<int>::iterator col_it;
    Matrix* epetra_matrix = MatrixCreate( raw_mpi_comm, beg_row,end_row );
    for ( int i = beg_row; i < end_row+1; ++i )
    {
        // Get the Epetra row.
	MCLS_CHECK( d_A->RowMatrixRowMap().MyGID(i) );
	local_row = d_A->RowMatrixRowMap().LID(i);
	error = d_A->ExtractMyRowCopy( local_row, 
                                       Teuchos::as<int>(values.size()), 
                                       num_entries,
                                       values.getRawPtr(), 
                                       indices.getRawPtr() );
        MCLS_CHECK( 0 == error );
	for ( col_it = indices.begin(); 
              col_it != indices.begin()+num_entries; 
              ++col_it )
	{
	    *col_it = d_A->RowMatrixColMap().GID(*col_it);
	}

        // Insert it into the Epetra ParaSails matrix.
        MatrixSetRow( epetra_matrix, i, num_entries, 
                      indices.getRawPtr(), values.getRawPtr() );
    }
    values.clear();
    indices.clear();

    // Fill Complete the Epetra ParaSails matrix.
    MatrixComplete( epetra_matrix );

    // Create a ParaSails preconditioner.
    ParaSails* parasails = ParaSailsCreate( raw_mpi_comm, beg_row, end_row, 0 );
    ParaSailsSetupPattern( parasails, epetra_matrix, threshold, num_levels );
    ParaSailsSetupValues( parasails, epetra_matrix, filter );

    // Destroy the ParaSails copy of the operator.
    MatrixDestroy( epetra_matrix );

    // Extract the ParaSails preconditioner into the Epetra preconditioner.
    d_preconditioner = Teuchos::rcp( 
	new Epetra_CrsMatrix(Copy,d_A->RowMatrixRowMap(),0) );
    Teuchos::ArrayView<int> mlens( parasails->M->lens, end_row-beg_row+1 );
    int max_m_entries = *std::max( mlens.begin(), mlens.end() );
    int num_m_entries = 0;
    Teuchos::ArrayRCP<double> m_values( max_m_entries );
    Teuchos::ArrayRCP<int> m_indices( max_m_entries );
    int* m_indices_ptr = m_indices.getRawPtr();
    double* m_values_ptr = m_values.getRawPtr();
    for ( int i = beg_row; i < end_row+1; ++i )
    {
        MatrixGetRow( parasails->M, i, &num_m_entries, 
                      &m_indices_ptr, &m_values_ptr );
        error = d_preconditioner->InsertGlobalValues(
            i, num_m_entries, m_values_ptr, m_indices_ptr );
        MCLS_CHECK( 0 == error );
    }
	    
    // ParaSails cleanup.
    ParaSailsDestroy( parasails );

    // Finalize.
    error = d_preconditioner->FillComplete();
    MCLS_CHECK( 0 == error );
	
    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
    MCLS_ENSURE( d_preconditioner->Filled() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraParaSailsPreconditioner.cpp
//---------------------------------------------------------------------------//
