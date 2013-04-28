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
 * \file MCLS_EpetraPSILUTPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief ParaSails ILUT preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include "MCLS_EpetraPSILUTPreconditioner.hpp"
#include "MCLS_EpetraRowMatrixAdapter.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>

#include <Epetra_Vector.h>

#include <Ifpack_ILUT.h>

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
EpetraPSILUTPreconditioner::EpetraPSILUTPreconditioner(
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
EpetraPSILUTPreconditioner::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<double>("fact: ilut level-of-fill", 1.0);
    plist->set<double>("fact: drop tolerance", 1.0e-2);
    plist->set<double>("fact: absolute threshold", 1.0 );
    plist->set<double>("fact: relative threshold", 1.0 );
    plist->set<double>("fact: relax value", 1.0 );
    plist->set<double>("ParaSails: Threshold", 0.0);
    plist->set<int>("ParaSails: Number of Levels", 0.0);
    plist->set<double>("ParaSails: Filter", 0.0);
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraPSILUTPreconditioner::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraPSILUTPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    MCLS_REQUIRE( Teuchos::nonnull(params) );
    d_plist = params;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraPSILUTPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Build the preconditioner.
 */
void EpetraPSILUTPreconditioner::buildPreconditioner()
{
    typedef MatrixTraits<Epetra_Vector,Epetra_RowMatrix> MT;

    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    std::cout << "MCLS PSILUT: Generating ILUT Factorization" << std::endl;
    Teuchos::Time timer("");
    timer.start(true);

    // Build the Ifpack ILUT preconditioner.
    Teuchos::RCP<Epetra_CrsMatrix> l_inv;
    {
        Ifpack_ILUT ifpack( d_A.getRawPtr() );
        int error = ifpack.SetParameters( *d_plist );
        MCLS_CHECK( 0 == error );
        error = ifpack.Initialize();
        MCLS_CHECK( 0 == error );
        MCLS_CHECK( ifpack.IsInitialized() );
        error = ifpack.Compute();
        MCLS_CHECK( 0 == error );
        MCLS_CHECK( ifpack.IsComputed() );

        // Invert L and U.
        std::cout << "MCLS PSILUT: Inverting ILUT Factorization" << std::endl;
        l_inv = computeTriInverse( ifpack.L(), d_A->RowMatrixRowMap(), false );
        d_u_inv = computeTriInverse( ifpack.U(), d_A->RowMatrixRowMap(), true );

        MCLS_CHECK( Teuchos::nonnull(l_inv) );
        MCLS_CHECK( Teuchos::nonnull(d_u_inv) );
    }

    std::cout << "MCLS PSILUT: Building Composite Operator" << std::endl;

    // Build the composite ILUT preconditioned operator.
    Teuchos::RCP<Epetra_RowMatrix> composite;
    {
        Teuchos::RCP<Epetra_RowMatrix> temp = MT::clone( *d_A );
        MT::multiply( d_A, d_u_inv, temp, false );
        composite = MT::clone( *l_inv );
        MT::multiply( l_inv, temp, composite, false );
    }

    std::cout << "MCLS PSILUT: Computing SPAINV Preconditioner" << std::endl;

    // Build the sparse approximate inverse preconditioner.
    Teuchos::RCP<Epetra_CrsMatrix> spainv = computeSparseInverse( composite );
    composite = Teuchos::null;

    std::cout << "MCLS PSILUT: Building Composite Left Preconditioner" << std::endl;

    // Build the composite left preconditioner.
    d_m_inv = MT::clone( *spainv );
    MT::multiply( spainv, l_inv, d_m_inv, false );

    timer.stop();
    std::cout << "MCLS PSILUT: Complete in " << timer.totalElapsedTime() 
              << " seconds." << std::endl;

    MCLS_ENSURE( Teuchos::nonnull(d_m_inv) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the inverse of a triangular matrix from Ifpack.
 */
Teuchos::RCP<Epetra_CrsMatrix>
EpetraPSILUTPreconditioner::computeTriInverse( const Epetra_CrsMatrix& A,
                                               const Epetra_Map& prec_map,
                                               bool is_upper )
{
    Teuchos::RCP<Epetra_CrsMatrix> inverse = Teuchos::rcp(
        new Epetra_CrsMatrix(Copy, prec_map, 0) );

    int num_rows = A.NumMyRows();
    Epetra_Vector basis( prec_map );
    Epetra_Vector inverse_row( prec_map );
    Teuchos::Array<double> values;
    Teuchos::Array<int> indices;
    double drop_tol = 0.0;
    if ( d_plist->isParameter("fact: drop tolerance") )
    {
        drop_tol = d_plist->get<double>("fact: drop tolerance");
    }

    // Invert the matrix row-by-row.
    int error = 0;
    for ( int i = 0; i < num_rows; ++i )
    {
        // Set the basis for this row.
        basis.PutScalar(0.0);
        basis[i] = 1.0;
            
        // Get the row for the inverse.
        error = A.Solve( is_upper, true, false, basis, inverse_row );
        MCLS_CHECK( 0 == error );

        // Get the non-zero elements of the row larger than the drop
        // tolerance.
        for ( int j = 0; j < num_rows; ++j )
        {
            if ( std::abs(inverse_row[j]) > drop_tol )
            {
                values.push_back( inverse_row[j] );
                indices.push_back( prec_map.GID(j) );
            }           
        }

        // Populate the row in the inverse matrix.
        error = inverse->InsertGlobalValues( prec_map.GID(i),
                                             values.size(),
                                             values.getRawPtr(),
                                             indices.getRawPtr() );
        MCLS_CHECK( 0 == error );

        values.clear();
        indices.clear();
    }

    error = inverse->FillComplete();
    MCLS_CHECK( 0 == error );

    return inverse;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the sparse approximate inverse of a matrix.
 */
Teuchos::RCP<Epetra_CrsMatrix>
EpetraPSILUTPreconditioner::computeSparseInverse( 
    const Teuchos::RCP<Epetra_RowMatrix>& A )
{
    // Build the sparse approximate inverse preconditioner.
    // Get the ParaSails parameters.
    double threshold = d_plist->get<double>("ParaSails: Threshold");
    int num_levels = d_plist->get<int>("ParaSails: Number of Levels");
    double filter = d_plist->get<double>("ParaSails: Filter");

    // Extract the raw MPI handle.
    Teuchos::RCP<const Epetra_Comm> epetra_comm = 
        Teuchos::rcp( &(A->Comm()), false );
    Teuchos::RCP<const Epetra_MpiComm> mpi_epetra_comm =
        Teuchos::rcp_dynamic_cast<const Epetra_MpiComm>( epetra_comm );
    MPI_Comm raw_mpi_comm = mpi_epetra_comm->Comm();

    // Export the operator to a row decomposition that is globally
    // contiguous. ParaSails requires this unfortunately.
    int error = 0;
    Epetra_Map linear_map( A->NumGlobalRows(), 0, A->Comm() );
    Teuchos::RCP<Epetra_CrsMatrix> contiguous_A = Teuchos::rcp( 
        new Epetra_CrsMatrix(Copy,linear_map,A->MaxNumEntries()) );
    Epetra_Export linear_export( A->RowMatrixRowMap(), linear_map );
    error = contiguous_A->Export( *A, linear_export, Insert );
    MCLS_CHECK( 0 == error );
    error = contiguous_A->FillComplete();
    MCLS_CHECK( 0 == error );
    MCLS_CHECK( contiguous_A->Filled() );

    // Check that the global ids are contiguous in the new operator.
    MCLS_CHECK( contiguous_A->RowMatrixRowMap().LinearMap() );
    MCLS_CHECK( contiguous_A->MaxNumEntries() > 0 );

    // Create a ParaSails matrix from the row-contiguous operator. 
    Teuchos::ArrayRCP<double> values( contiguous_A->MaxNumEntries() );
    Teuchos::ArrayRCP<double>::iterator values_it;
    Teuchos::ArrayRCP<int> indices( contiguous_A->MaxNumEntries() );
    Teuchos::ArrayRCP<int>::iterator indices_it;
    int num_entries = 0;
    int local_row = 0;
    int beg_row = contiguous_A->RowMatrixRowMap().MinMyGID();
    int end_row = contiguous_A->RowMatrixRowMap().MaxMyGID();
    Teuchos::ArrayRCP<int>::iterator col_it, col_it_2;
    Matrix* epetra_matrix = MatrixCreate( raw_mpi_comm, beg_row, end_row );
    for ( int i = beg_row; i < end_row+1; ++i )
    {
        // Get the Epetra row.
	MCLS_CHECK( contiguous_A->RowMatrixRowMap().MyGID(i) );
	local_row = contiguous_A->RowMatrixRowMap().LID(i);

	error = contiguous_A->ExtractMyRowCopy( local_row, 
                                                Teuchos::as<int>(values.size()), 
                                                num_entries,
                                                values.getRawPtr(), 
                                                indices.getRawPtr() );
        MCLS_CHECK( 0 == error );
        MCLS_CHECK( num_entries > 0 );

        // Get rid of the zero entries.
        double tol = 1.0e-15;
        for ( values_it = values.begin(), indices_it = indices.begin();
              values_it != values.begin()+num_entries;
              ++values_it, ++indices_it )
        {
            if ( std::abs(*values_it) < tol )
            {
                *values_it = 0.0;
                *indices_it = -1;
            }
        }
        values_it = 
            std::remove( values.begin(), values.begin()+num_entries, 0.0 );
        indices_it = 
            std::remove( indices.begin(), indices.begin()+num_entries, -1 );

        // Convert the local indices into global indices.
	for ( col_it = indices.begin(); 
              col_it != indices_it; 
              ++col_it )
	{
	    *col_it = contiguous_A->RowMatrixColMap().GID(*col_it);
	}

        // Insert it into the Epetra ParaSails matrix.
        MCLS_CHECK( A->Comm().MyPID() == MatrixRowPe(epetra_matrix, i) );
        num_entries = std::distance( values.begin(), values_it );
        MatrixSetRow( epetra_matrix, i, num_entries, 
                      indices.getRawPtr(), values.getRawPtr() );
    }
    values.clear();
    indices.clear();

    // Free the contiguous copy of the operator.
    contiguous_A = Teuchos::null;

    // Fill Complete the Epetra ParaSails matrix.
    MatrixComplete( epetra_matrix );

    // Create a ParaSails preconditioner.
    ParaSails* parasails = ParaSailsCreate( raw_mpi_comm, beg_row, end_row, 0 );
    ParaSailsSetupPattern( parasails, epetra_matrix, threshold, num_levels );
    ParaSailsSetupValues( parasails, epetra_matrix, filter );

    // Parasails timing output - only if DBC is enabled.
    MCLS_REMEMBER( std::cout << std::endl );
    MCLS_REMEMBER( ParaSailsStatsPattern(parasails, epetra_matrix) );
    MCLS_REMEMBER( ParaSailsStatsValues(parasails, epetra_matrix) );

    // Destroy the ParaSails copy of the operator.
    MatrixDestroy( epetra_matrix );

    // Build a contiguous preconditioner.
    Teuchos::ArrayView<int> mlens( parasails->M->lens, end_row-beg_row+1 );
    int max_m_entries = *std::max_element( mlens.begin(), mlens.end() );
    MCLS_CHECK( max_m_entries > 0 );
    Teuchos::RCP<Epetra_CrsMatrix> contiguous_M = Teuchos::rcp( 
        new Epetra_CrsMatrix(Copy,linear_map,max_m_entries) );

    // Extract the ParaSails preconditioner into the contiguous preconditioner.
    int num_m_entries = 0;
    Teuchos::Array<int> global_indices;
    int* m_indices_ptr;
    double* m_values_ptr;
    for ( int i = beg_row; i < end_row+1; ++i )
    {
        local_row = i-beg_row;
        MCLS_CHECK( A->Comm().MyPID() == MatrixRowPe(parasails->M, i) );
        MatrixGetRow( parasails->M, local_row, &num_m_entries, 
                      &m_indices_ptr, &m_values_ptr );

        global_indices.resize( num_m_entries );
        NumberingLocalToGlobal( parasails->M->numb, num_m_entries,
                                m_indices_ptr, global_indices.getRawPtr() );

        MCLS_CHECK( contiguous_M->RowMatrixRowMap().MyGID(i) );
        error = contiguous_M->InsertGlobalValues(
            i, num_m_entries, m_values_ptr, global_indices.getRawPtr() );
        MCLS_CHECK( 0 == error );
    }
    global_indices.clear();
    
    // Barrier before continuing.
    A->Comm().Barrier();

    // ParaSails cleanup.
    ParaSailsDestroy( parasails );

    // Finalize extracted inverse.
    error = contiguous_M->FillComplete();
    MCLS_CHECK( 0 == error );
    MCLS_CHECK( contiguous_M->Filled() );

    // Export the contiguous preconditioner into the operator decomposition.
    Teuchos::RCP<Epetra_CrsMatrix> ainv_prec = Teuchos::rcp(
	new Epetra_CrsMatrix(Copy,A->RowMatrixRowMap(),max_m_entries) );
    Epetra_Export base_export( linear_map, A->RowMatrixRowMap() );
    error = ainv_prec->Export( *contiguous_M, base_export, Insert );
    MCLS_CHECK( 0 == error );

    // Free the contiguous copy of the preconditioner.
    contiguous_M = Teuchos::null;

    // Finalize the preconditioner.
    error = ainv_prec->FillComplete();
    MCLS_CHECK( 0 == error );

    MCLS_ENSURE( Teuchos::nonnull(ainv_prec) );
    MCLS_ENSURE( ainv_prec->Filled() );

    return ainv_prec;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraPSILUTPreconditioner.cpp
//---------------------------------------------------------------------------//
