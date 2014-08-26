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
 * \file MCLS_TpetraParaSailsPreconditioner_impl.hpp
 * \author Stuart R. Slattery
 * \brief ParaSails preconditioning for Tpetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAPARASAILS_IMPL_HPP
#define MCLS_TPETRAPARASAILS_IMPL_HPP

#include <MCLS_DBC.hpp>

#include <limits>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_Export.hpp>

#include <ParaSails.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Scalar, class LO, class GO>
TpetraParaSailsPreconditioner<Scalar,LO,GO>::TpetraParaSailsPreconditioner(
    const Teuchos::RCP<Teuchos::ParameterList>& params )
    : d_plist( params )
#if HAVE_MCLS_TIMERS
    , d_prec_timer( Teuchos::TimeMonitor::getNewCounter("ParaSails Create") )
#endif
{
    MCLS_REQUIRE( Teuchos::nonnull(d_plist) );
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the valid parameters for this preconditioner.
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<const Teuchos::ParameterList> 
TpetraParaSailsPreconditioner<Scalar,LO,GO>::getValidParameters() const
{
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<double>("ParaSails: Threshold", 0.0);
    plist->set<int>("ParaSails: Number of Levels", 0.0);
    plist->set<double>("ParaSails: Filter", 0.0);
    plist->set<int>("ParaSails: Symmetry", 0 );
    plist->set<int>("ParaSails: Load Balance", 1 );
    return plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<const Teuchos::ParameterList> 
TpetraParaSailsPreconditioner<Scalar,LO,GO>::getCurrentParameters() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
template<class Scalar, class LO, class GO>
void TpetraParaSailsPreconditioner<Scalar,LO,GO>::setParameters( 
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
void TpetraParaSailsPreconditioner<Scalar,LO,GO>::setOperator( 
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
void TpetraParaSailsPreconditioner<Scalar,LO,GO>::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->isFillComplete() );

#if HAVE_MCLS_TIMERS
    Teuchos::TimeMonitor prec_monitor( *d_prec_timer );
#endif

    // Get the ParaSails parameters.
    double threshold = d_plist->get<double>("ParaSails: Threshold");
    int num_levels = d_plist->get<int>("ParaSails: Number of Levels");
    double filter = d_plist->get<double>("ParaSails: Filter");
    int symmetry = d_plist->get<int>("ParaSails: Symmetry");
    int load_balance = d_plist->get<int>("ParaSails: Load Balance");

    // Extract the raw MPI handle.
    Teuchos::RCP<const Teuchos::Comm<int> > comm = d_A->getComm();
    Teuchos::RCP< const Teuchos::MpiComm<int> > mpi_comm = 
	Teuchos::rcp_dynamic_cast< const Teuchos::MpiComm<int> >( comm );
    Teuchos::RCP< const Teuchos::OpaqueWrapper<MPI_Comm> > opaque_comm = 
	mpi_comm->getRawMpiComm();
    MPI_Comm raw_mpi_comm = (*opaque_comm)();

    // Export the operator to a row decomposition that is globally
    // contiguous. ParaSails requires this unfortunately.
    Teuchos::RCP<const Tpetra::Map<LO,GO> > linear_map = 
	Tpetra::createUniformContigMap<LO,GO>( d_A->getGlobalNumRows(), comm );
    Teuchos::RCP<Tpetra::CrsMatrix<Scalar,LO,GO> > contiguous_A = 
	Tpetra::createCrsMatrix<Scalar,LO,GO>( 
	    linear_map, d_A->getGlobalMaxNumRowEntries() );
    Tpetra::Export<LO,GO> linear_export( d_A->getMap(), linear_map );
    contiguous_A->doExport( *d_A, linear_export, Tpetra::INSERT );
    contiguous_A->fillComplete();
    MCLS_CHECK( contiguous_A->isFillComplete() );

    // Check that the global ids are contiguous in the new operator.
    MCLS_CHECK( contiguous_A->getMap()->isContiguous() );
    MCLS_CHECK( contiguous_A->getGlobalMaxNumRowEntries() > 0 );

    // Create a ParaSails matrix from the row-contiguous operator.
    Teuchos::ArrayRCP<double> values( 
	contiguous_A->getGlobalMaxNumRowEntries() );
    Teuchos::ArrayRCP<double>::iterator values_it;
    Teuchos::ArrayRCP<GO> indices( contiguous_A->getGlobalMaxNumRowEntries() );
    typename Teuchos::ArrayRCP<GO>::iterator indices_it;
    std::size_t num_entries = 0;
    int beg_row = contiguous_A->getMap()->getMinGlobalIndex();
    int end_row = contiguous_A->getMap()->getMaxGlobalIndex();
    double tol = 10.0*std::numeric_limits<double>::epsilon();
    Teuchos::ArrayRCP<int>::iterator col_it, col_it_2;
    Matrix* tpetra_matrix = MatrixCreate( raw_mpi_comm, beg_row, end_row );
    for ( int i = beg_row; i < end_row+1; ++i )
    {
        // Get the Tpetra row.
	MCLS_CHECK( contiguous_A->getMap()->getLocalElement(i) !=
		    Teuchos::OrdinalTraits<LO>::invalid() );
	contiguous_A->getGlobalRowCopy( 
	    i, indices(), values(), num_entries );
        MCLS_CHECK( num_entries > 0 );

        // Get rid of the zero entries.
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

        // Insert it into the Tpetra ParaSails matrix.
        MCLS_CHECK( comm->getRank() == MatrixRowPe(tpetra_matrix,i) );
        num_entries = std::distance( values.begin(), values_it );
        MatrixSetRow( tpetra_matrix, i, num_entries, 
                      indices.getRawPtr(), values.getRawPtr() );
    }
    values.clear();
    indices.clear();

    // Free the contiguous copy of the operator.
    contiguous_A = Teuchos::null;

    // Fill Complete the Tpetra ParaSails matrix.
    MatrixComplete( tpetra_matrix );

    // Create a ParaSails preconditioner.
    ParaSails* parasails = 
	ParaSailsCreate( raw_mpi_comm, beg_row, end_row, symmetry );
    parasails->loadbal_beta = load_balance;
    ParaSailsSetupPattern( parasails, tpetra_matrix, threshold, num_levels );
    ParaSailsStatsPattern( parasails, tpetra_matrix );
    ParaSailsSetupValues( parasails, tpetra_matrix, filter );
    MCLS_REMEMBER( ParaSailsStatsValues(parasails, tpetra_matrix) );

    // Destroy the ParaSails copy of the operator.
    MatrixDestroy( tpetra_matrix );

    // Build a contiguous preconditioner.
    Teuchos::ArrayView<int> mlens( parasails->M->lens, end_row-beg_row+1 );
    int max_m_entries = *std::max_element( mlens.begin(), mlens.end() );
    MCLS_CHECK( max_m_entries > 0 );
    Teuchos::RCP<Tpetra::CrsMatrix<Scalar,LO,GO> > contiguous_M = 
	Tpetra::createCrsMatrix<Scalar,LO,GO>( linear_map, max_m_entries );

    // Extract the ParaSails preconditioner into the contiguous
    // preconditioner.
    int local_row = 0;
    int num_m_entries = 0;
    Teuchos::Array<GO> global_indices( max_m_entries );
    int* m_indices_ptr;
    double* m_values_ptr;
    for ( int i = beg_row; i < end_row+1; ++i )
    {
        local_row = i-beg_row;
        MCLS_CHECK( comm->getRank() == MatrixRowPe(parasails->M,i) );
        MatrixGetRow( parasails->M, local_row, &num_m_entries, 
                      &m_indices_ptr, &m_values_ptr );

        NumberingLocalToGlobal( parasails->M->numb, num_m_entries,
                                m_indices_ptr, global_indices.getRawPtr() );

        MCLS_CHECK( contiguous_M->getMap()->isNodeGlobalElement(i) );
        contiguous_M->insertGlobalValues(
            i, 
	    global_indices(0,num_m_entries),
	    Teuchos::ArrayView<Scalar>(m_values_ptr,num_m_entries) );
    }
    global_indices.clear();
    
    // Barrier before continuing.
    comm->barrier();

    // ParaSails cleanup.
    ParaSailsDestroy( parasails );

    // Finalize extracted inverse.
    contiguous_M->fillComplete();
    MCLS_CHECK( contiguous_M->isFillComplete() );

    // Export the contiguous preconditioner into the operator decomposition.
    d_preconditioner = 
	Tpetra::createCrsMatrix<Scalar,LO,GO>( d_A->getMap(), max_m_entries );
    Tpetra::Export<LO,GO> base_export( linear_map, d_A->getMap() );
    d_preconditioner->doExport( 
	*contiguous_M, base_export, Tpetra::INSERT );

    // Free the contiguous copy of the preconditioner.
    contiguous_M = Teuchos::null;

    // Finalize the preconditioner.
    d_preconditioner->fillComplete();

    MCLS_ENSURE( Teuchos::nonnull(d_preconditioner) );
    MCLS_ENSURE( d_preconditioner->isFillComplete() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAPARASAILS_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraParaSailsPreconditioner_impl.hpp
//---------------------------------------------------------------------------//
