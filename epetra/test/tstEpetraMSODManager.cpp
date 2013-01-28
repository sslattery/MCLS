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
 * \file tstEpetraMSODManager.cpp
 * \author Stuart R. Slattery
 * \brief Epetra MSOD manager tests.
 */
//---------------------------------------------------------------------------//

#include <stack>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_MSODManager.hpp>
#include <MCLS_UniformAdjointSource.hpp>
#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_History.hpp>
#include <MCLS_AdjointTally.hpp>
#include <MCLS_Events.hpp>
#include <MCLS_RNGControl.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//

Teuchos::RCP<Epetra_Comm> getEpetraComm( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& comm )
{
#ifdef HAVE_MPI
    Teuchos::RCP< const Teuchos::MpiComm<int> > mpi_comm = 
    	Teuchos::rcp_dynamic_cast< const Teuchos::MpiComm<int> >( comm );
    Teuchos::RCP< const Teuchos::OpaqueWrapper<MPI_Comm> > opaque_comm = 
    	mpi_comm->getRawMpiComm();
    return Teuchos::rcp( new Epetra_MpiComm( (*opaque_comm)() ) );
#else
    return Teuchos::rcp( new Epetra_SerialComm() );
#endif
}

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MSODManager, two_by_two )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::UniformAdjointSource<DomainType> SourceType;
    typedef DomainType::TallyType TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This is a 4 processor test.
    if ( comm_size == 4 )
    {

	// Build the set-constant communicator.
	Teuchos::Array<int> ranks(2);
	if ( comm_rank < 2 )
	{
	    ranks[0] = 0;
	    ranks[1] = 1;
	}
	else
	{
	    ranks[0] = 2;
	    ranks[1] = 3;
	}
	Teuchos::RCP<const Teuchos::Comm<int> > comm_set =
	    comm->createSubcommunicator( ranks() );
	int set_size = comm_set->getSize();
	int set_rank = comm_set->getRank();

	// Build the block-constant communicator.
	if ( comm_rank == 0 || comm_rank == 2 )
	{
	    ranks[0] = 0;
	    ranks[1] = 2;
	}
	else
	{
	    ranks[0] = 1;
	    ranks[1] = 3;
	}
	Teuchos::RCP<const Teuchos::Comm<int> > comm_block =
	    comm->createSubcommunicator( ranks() );

	// History and RNG setup.
	Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	    new MCLS::RNGControl( 3939294 ) );
	HistoryType::setByteSize( control->getSize() );

	// Parameters.
	double cutoff = 1.0e-8;
	Teuchos::ParameterList plist;
	plist.set<int>( "Overlap Size", 2 );
	plist.set<double>( "Weight Cutoff", cutoff );
	plist.set<int>( "Number of Sets", 2 );

	// Declare variables globally.
	Teuchos::RCP<DomainType> primary_domain;
	Teuchos::RCP<SourceType> primary_source;

	// Linear system sizes.
	int local_num_rows = 10;
	int global_num_rows = local_num_rows*set_size;

	// Create the solution vector externally so that we can check it.
	Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm_set );
	Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	    new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

	Teuchos::RCP<VectorType> x = Teuchos::rcp( new Epetra_Vector(*map) );

	// Build the primary source and domain on set 0.
	if ( comm_rank < 2 )
	{
	    // Build the linear operator and solution vector.
	    Teuchos::RCP<Epetra_CrsMatrix> A = 
		Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
	    Teuchos::Array<int> global_columns( 2 );
	    Teuchos::Array<double> values( 2 );
	    for ( int i = 1; i < global_num_rows; ++i )
	    {
		global_columns[0] = i-1;
		global_columns[1] = i;
		values[0] = 2;
		values[1] = 3;
		A->InsertGlobalValues( i, global_columns.size(), 
				       &values[0], &global_columns[0] );
	    }
	    A->FillComplete();

	    // Build the primary adjoint domain.
	    Teuchos::RCP<MatrixType> B = A;
	    primary_domain = Teuchos::rcp( new DomainType(B, x, plist) );

	    // Create the primary adjoint source with default values.
	    Teuchos::RCP<VectorType> b = VT::clone( *x );
	    VT::putScalar( *b, -1.0 );
	    primary_source = Teuchos::rcp( 
		new SourceType( b, primary_domain, control, comm_set, 
				comm_rank, comm_size, plist ) );
	}
	comm->barrier();

	// Build the MSOD manager.
	MCLS::MSODManager<DomainType,SourceType> msod_manager( primary_domain,
							       primary_source,
							       comm,
							       control,
							       plist );

	// Test the MSOD manager.
	TEST_EQUALITY( msod_manager.numSets(), 2 );
	TEST_EQUALITY( msod_manager.numBlocks(), 2 );
	TEST_EQUALITY( msod_manager.setSize(), 2 );
	TEST_EQUALITY( msod_manager.blockSize(), 2 );
	TEST_EQUALITY( msod_manager.setComm()->getSize(), comm_set->getSize() );
	TEST_EQUALITY( msod_manager.setComm()->getRank(), comm_set->getRank() );
	TEST_EQUALITY( msod_manager.blockComm()->getSize(), comm_block->getSize() );
	TEST_EQUALITY( msod_manager.blockComm()->getRank(), comm_block->getRank() );
	if ( comm_rank == 0 )
	{
	    TEST_EQUALITY( msod_manager.setID(), 0 );
	    TEST_EQUALITY( msod_manager.blockID(), 0 );
	}
	else if ( comm_rank == 1 )
	{
	    TEST_EQUALITY( msod_manager.setID(), 0 );
	    TEST_EQUALITY( msod_manager.blockID(), 1 );
	}
	else if ( comm_rank == 2 )
	{
	    TEST_EQUALITY( msod_manager.setID(), 1 );
	    TEST_EQUALITY( msod_manager.blockID(), 0 );
	}
	else
	{
	    TEST_EQUALITY( msod_manager.setID(), 1 );
	    TEST_EQUALITY( msod_manager.blockID(), 1 );
	}

	// Test the local domain.
	Teuchos::RCP<DomainType> domain = msod_manager.localDomain();

	// Check the tally.
	double x_val = 2;
	Teuchos::RCP<TallyType> tally = domain->domainTally();
	tally->setBaseVector( x );
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*set_rank && i < 2+local_num_rows*(set_rank+1) )
	    {
		HistoryType history( i, x_val );
		history.live();
		tally->tallyHistory( history );
	    }
	}

	tally->combineSetTallies();

	Teuchos::ArrayRCP<const double> x_view = VT::view( *x );
	Teuchos::ArrayRCP<const double>::const_iterator x_view_iterator;
	for ( x_view_iterator = x_view.begin();
	      x_view_iterator != x_view.end();
	      ++x_view_iterator )
	{
	    TEST_EQUALITY( *x_view_iterator, x_val );
	}

	// Check the boundary.
	if ( set_rank == 0 && set_size == 1 )
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 0 );
	}
	else if ( set_rank == 0 && set_size > 1 )
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 1 );
	    TEST_EQUALITY( domain->sendNeighborRank(0), set_rank+1 );
	    TEST_EQUALITY( domain->owningNeighbor(2+local_num_rows*(set_rank+1)), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 0 );
	}
	else if ( set_rank == set_size - 1 )
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 1 );
	    TEST_EQUALITY( domain->receiveNeighborRank(0), set_rank-1 );
	}
	else
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 1 );
	    TEST_EQUALITY( domain->sendNeighborRank(0), set_rank+1 );
	    TEST_EQUALITY( domain->owningNeighbor(2*local_num_rows*(set_rank+1)), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 1 );
	    TEST_EQUALITY( domain->receiveNeighborRank(0), set_rank-1 );
	}

	if ( set_rank == set_size-1 )
	{
	    for ( int i = 0; i < global_num_rows; ++i )
	    {
		if ( i >= local_num_rows*set_rank && 
		     i < local_num_rows*(set_rank+1) )
		{
		    TEST_ASSERT( domain->isLocalState(i) );
		}
		else
		{
		    TEST_ASSERT( !domain->isLocalState(i) );
		}
	    }
	}
	else
	{
	    for ( int i = 0; i < global_num_rows; ++i )
	    {
		if ( i >= local_num_rows*set_rank && 
		     i < 2+local_num_rows*(set_rank+1) )
		{
		    TEST_ASSERT( domain->isLocalState(i) );
		}
		else
		{
		    TEST_ASSERT( !domain->isLocalState(i) );
		}
	    }
	}

	// Test the source.
	Teuchos::RCP<SourceType> source = msod_manager.localSource();

	TEST_ASSERT( source->empty() );
	TEST_EQUALITY( source->numToTransport(), 0 );
	TEST_EQUALITY( source->numToTransportInSet(), global_num_rows );
	TEST_EQUALITY( source->numRequested(), global_num_rows );
	TEST_EQUALITY( source->numLeft(), 0 );
	TEST_EQUALITY( source->numEmitted(), 0 );
	TEST_EQUALITY( source->numStreams(), 0 );

	// Build the source.
	source->buildSource();
	TEST_ASSERT( !source->empty() );
	TEST_EQUALITY( source->numToTransport(), local_num_rows );
	TEST_EQUALITY( source->numToTransportInSet(), global_num_rows );
	TEST_EQUALITY( source->numRequested(), global_num_rows );
	TEST_EQUALITY( source->numLeft(), local_num_rows );
	TEST_EQUALITY( source->numEmitted(), 0 );
	TEST_EQUALITY( source->numStreams(), comm->getSize() );

	// Sample the source.
	for ( int i = 0; i < local_num_rows; ++i )
	{
	    TEST_ASSERT( !source->empty() );
	    TEST_EQUALITY( source->numLeft(), local_num_rows-i );
	    TEST_EQUALITY( source->numEmitted(), i );

	    Teuchos::RCP<HistoryType> history = source->getHistory();

	    TEST_EQUALITY( history->weight(), -global_num_rows );
	    TEST_ASSERT( domain->isLocalState( history->state() ) );
	    TEST_ASSERT( history->alive() );
	    TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
	}
	TEST_ASSERT( source->empty() );
	TEST_EQUALITY( source->numLeft(), 0 );
	TEST_EQUALITY( source->numEmitted(), local_num_rows );

	// Now update the domain and source in the manager and check again.
	domain = Teuchos::null;
	source = Teuchos::null;

	msod_manager.updateDomain( primary_domain );
	msod_manager.updateSource( primary_source );

	// Test the local domain.
	domain = msod_manager.localDomain();

	// Reset and check the tally.
	x_val = 2;
	tally = domain->domainTally();
	VT::putScalar( *x, 0 );
	tally->setBaseVector( x );
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*set_rank && i < 2+local_num_rows*(set_rank+1) )
	    {
		HistoryType history( i, x_val );
		history.live();
		tally->tallyHistory( history );
	    }
	}

	tally->combineSetTallies();

	x_view = VT::view( *x );
	for ( x_view_iterator = x_view.begin();
	      x_view_iterator != x_view.end();
	      ++x_view_iterator )
	{
	    TEST_EQUALITY( *x_view_iterator, x_val );
	}

	// Check the boundary.
	if ( set_rank == 0 && set_size == 1 )
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 0 );
	}
	else if ( set_rank == 0 && set_size > 1 )
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 1 );
	    TEST_EQUALITY( domain->sendNeighborRank(0), set_rank+1 );
	    TEST_EQUALITY( domain->owningNeighbor(2+local_num_rows*(set_rank+1)), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 0 );
	}
	else if ( set_rank == set_size - 1 )
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 1 );
	    TEST_EQUALITY( domain->receiveNeighborRank(0), set_rank-1 );
	}
	else
	{
	    TEST_EQUALITY( domain->numSendNeighbors(), 1 );
	    TEST_EQUALITY( domain->sendNeighborRank(0), set_rank+1 );
	    TEST_EQUALITY( domain->owningNeighbor(2*local_num_rows*(set_rank+1)), 0 );
	    TEST_EQUALITY( domain->numReceiveNeighbors(), 1 );
	    TEST_EQUALITY( domain->receiveNeighborRank(0), set_rank-1 );
	}

	if ( set_rank == set_size-1 )
	{
	    for ( int i = 0; i < global_num_rows; ++i )
	    {
		if ( i >= local_num_rows*set_rank && 
		     i < local_num_rows*(set_rank+1) )
		{
		    TEST_ASSERT( domain->isLocalState(i) );
		}
		else
		{
		    TEST_ASSERT( !domain->isLocalState(i) );
		}
	    }
	}
	else
	{
	    for ( int i = 0; i < global_num_rows; ++i )
	    {
		if ( i >= local_num_rows*set_rank && 
		     i < 2+local_num_rows*(set_rank+1) )
		{
		    TEST_ASSERT( domain->isLocalState(i) );
		}
		else
		{
		    TEST_ASSERT( !domain->isLocalState(i) );
		}
	    }
	}

	// Test the source.
	source = msod_manager.localSource();

	TEST_ASSERT( source->empty() );
	TEST_EQUALITY( source->numToTransport(), 0 );
	TEST_EQUALITY( source->numToTransportInSet(), global_num_rows );
	TEST_EQUALITY( source->numRequested(), global_num_rows );
	TEST_EQUALITY( source->numLeft(), 0 );
	TEST_EQUALITY( source->numEmitted(), 0 );
	TEST_EQUALITY( source->numStreams(), 0 );

	// Build the source.
	source->buildSource();
	TEST_ASSERT( !source->empty() );
	TEST_EQUALITY( source->numToTransport(), local_num_rows );
	TEST_EQUALITY( source->numToTransportInSet(), global_num_rows );
	TEST_EQUALITY( source->numRequested(), global_num_rows );
	TEST_EQUALITY( source->numLeft(), local_num_rows );
	TEST_EQUALITY( source->numEmitted(), 0 );
	TEST_EQUALITY( source->numStreams(), comm->getSize() );

	// Sample the source.
	for ( int i = 0; i < local_num_rows; ++i )
	{
	    TEST_ASSERT( !source->empty() );
	    TEST_EQUALITY( source->numLeft(), local_num_rows-i );
	    TEST_EQUALITY( source->numEmitted(), i );

	    Teuchos::RCP<HistoryType> history = source->getHistory();

	    TEST_EQUALITY( history->weight(), -global_num_rows );
	    TEST_ASSERT( domain->isLocalState( history->state() ) );
	    TEST_ASSERT( history->alive() );
	    TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
	}
	TEST_ASSERT( source->empty() );
	TEST_EQUALITY( source->numLeft(), 0 );
	TEST_EQUALITY( source->numEmitted(), local_num_rows );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraMSODManager.cpp
//---------------------------------------------------------------------------//

