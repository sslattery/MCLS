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
 * \file tstTpetraDomainCommunicator.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra DomainCommunicator tests.
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

#include <MCLS_DomainCommunicator.hpp>
#include <MCLS_DomainTransporter.hpp>
#include <MCLS_AlmostOptimalDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_AdjointHistory.hpp>
#include <MCLS_AdjointTally.hpp>
#include <MCLS_Events.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//
MCLS::AdjointHistory<long> makeHistory( 
    long state, double weight, int streamid )
{
    MCLS::AdjointHistory<long> history( state, state, weight );
    history.setEvent( MCLS::Event::BOUNDARY );
    return history;
}

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainCommunicator, Communicate )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef Tpetra::CrsMatrix<double,int,long> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<long> HistoryType;
    typedef std::mt19937 rng_type;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AlmostOptimalDomain<VectorType,MatrixType,rng_type,TallyType>
	DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This test is parallel.
    if ( comm_size > 1 )
    {
	int local_num_rows = 10;
	int global_num_rows = local_num_rows*comm_size;
	Teuchos::RCP<const Tpetra::Map<int,long> > map = 
	    Tpetra::createUniformContigMap<int,long>( global_num_rows, comm );

	// Build the linear operator and solution vector.
	Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<double,int,long>( map );
	Teuchos::Array<long> global_columns( 1 );
	Teuchos::Array<double> values( 1 );
	for ( int i = 1; i < global_num_rows; ++i )
	{
	    global_columns[0] = i-1;
	    values[0] = -0.5/comm_size;
	    A->insertGlobalValues( i, global_columns(), values() );
	}
	global_columns[0] = global_num_rows-1;
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
	A->fillComplete();

	Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
        Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

	// Build the adjoint domain.
	Teuchos::ParameterList plist;
	plist.set<int>( "Overlap Size", 0 );
	Teuchos::RCP<DomainType> domain = 
            Teuchos::rcp( new DomainType( A_T, x, plist ) );
	Teuchos::RCP<MCLS::PRNG<rng_type> > rng = Teuchos::rcp(
	    new MCLS::PRNG<rng_type>( comm->getRank() ) );
	domain->setRNG( rng );

	// History setup.
	HistoryType::setByteSize();

	// Build the domain communicator.
	typename MCLS::DomainCommunicator<DomainType>::BankType bank;
	int buffer_size = 3;
	plist.set<int>( "MC Buffer Size", buffer_size );
	MCLS::DomainCommunicator<DomainType> communicator( domain, comm, plist );

	// Test initialization.
	TEST_EQUALITY( Teuchos::as<int>(communicator.maxBufferSize()), buffer_size );
	TEST_ASSERT( !communicator.sendStatus() );
	TEST_ASSERT( !communicator.receiveStatus() );

	// Post receives.
	communicator.post();
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}

	// End communication.
	communicator.end();
	TEST_ASSERT( !communicator.receiveStatus() );

	// Post new receives.
	communicator.post();
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}
	TEST_EQUALITY( communicator.sendBufferSize(), 0 );

	// Flush with zero histories.
	TEST_EQUALITY( communicator.flush(), 0 );
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}
	TEST_ASSERT( !communicator.sendStatus() );

	// Receive empty flushed buffers.
	int zero_histories = communicator.wait( bank );
	TEST_EQUALITY( zero_histories, 0 );
	TEST_ASSERT( !communicator.receiveStatus() );
	TEST_ASSERT( bank.empty() );

	// Repost receives.
	communicator.post();
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !communicator.receiveStatus() );
	}
	else
	{
	    TEST_ASSERT( communicator.receiveStatus() );
	}

	// Proc 0 will send to proc 1.
	if ( comm_rank == 0 )
	{
	    TEST_ASSERT( !domain->isGlobalState(10) );

	    HistoryType h1 = 
		makeHistory( 10, 1.1, comm_rank*4 + 1 );
	    const typename MCLS::DomainCommunicator<DomainType>::Result
		r1 = communicator.communicate( h1 );
	    TEST_ASSERT( !r1.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 1 );

	    HistoryType h2 = 
		makeHistory( 10, 2.1, comm_rank*4 + 2 );
	    const typename MCLS::DomainCommunicator<DomainType>::Result
		r2 = communicator.communicate( h2 );
	    TEST_ASSERT( !r2.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 2 );

	    HistoryType h3 = 
		makeHistory( 10, 3.1, comm_rank*4 + 3 );
	    const typename MCLS::DomainCommunicator<DomainType>::Result
		r3 = communicator.communicate( h3 );
	    TEST_ASSERT( r3.sent );
	    TEST_EQUALITY( r3.destination, 1 );
	    TEST_EQUALITY( communicator.sendBufferSize(), 0 );
	}

	// Proc comm_rank send to proc comm_rank+1 and receive from proc
	// comm_rank-1. 
	else if ( comm_rank < comm_size - 1 )
	{
	    // Send to proc comm_rank+1.
	    TEST_ASSERT( !domain->isGlobalState((comm_rank+1)*10) );

	    HistoryType h1 = 
		makeHistory( (comm_rank+1)*10, 1.1, comm_rank*4 + 1 );
	    const typename MCLS::DomainCommunicator<DomainType>::Result
		r1 = communicator.communicate( h1 );
	    TEST_ASSERT( !r1.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 1 );

	    HistoryType h2 = 
		makeHistory( (comm_rank+1)*10, 2.1, comm_rank*4 + 2 );
	    const typename MCLS::DomainCommunicator<DomainType>::Result
		r2 = communicator.communicate( h2 );
	    TEST_ASSERT( !r2.sent );
	    TEST_EQUALITY( communicator.sendBufferSize(), 2 );

	    HistoryType h3 = 
		makeHistory( (comm_rank+1)*10, 3.1, comm_rank*4 + 3 );
	    const typename MCLS::DomainCommunicator<DomainType>::Result
		r3 = communicator.communicate( h3 );
	    TEST_ASSERT( r3.sent );
	    TEST_EQUALITY( r3.destination, comm_rank+1 );
	    TEST_EQUALITY( communicator.sendBufferSize(), 0 );

	    // Receive from proc comm_rank-1.
	    while ( bank.empty() )
	    {
		communicator.checkAndPost( bank );
	    }

	    TEST_EQUALITY( bank.size(), 3 );

	    HistoryType rp3 = bank.top();
	    bank.pop();
	    HistoryType rp2 = bank.top();
	    bank.pop();
	    HistoryType rp1 = bank.top();
	    bank.pop();
	    TEST_ASSERT( bank.empty() );

	    TEST_EQUALITY( rp3.globalState(), comm_rank*10 );
	    TEST_EQUALITY( rp3.weight(), 3.1 );
	    TEST_EQUALITY( rp2.globalState(), comm_rank*10 );
	    TEST_EQUALITY( rp2.weight(), 2.1 );
	    TEST_EQUALITY( rp1.globalState(), comm_rank*10 );
	    TEST_EQUALITY( rp1.weight(), 1.1 );

	    TEST_ASSERT( communicator.receiveStatus() );
	}

	// The last proc just receives.
	else
	{
	    // Check and post until receive from proc comm_rank-1
	    while ( bank.empty() )
	    {
		communicator.checkAndPost( bank );
	    }
	    TEST_ASSERT( communicator.receiveStatus() );
	    TEST_EQUALITY( bank.size(), 3 );

	    HistoryType rp3 = bank.top();
	    bank.pop();
	    HistoryType rp2 = bank.top();
	    bank.pop();
	    HistoryType rp1 = bank.top();
	    bank.pop();
	    TEST_ASSERT( bank.empty() );

	    TEST_EQUALITY( rp3.globalState(), comm_rank*10 );
	    TEST_EQUALITY( rp3.weight(), 3.1 );
	    TEST_EQUALITY( rp2.globalState(), comm_rank*10 );
	    TEST_EQUALITY( rp2.weight(), 2.1 );
	    TEST_EQUALITY( rp1.globalState(), comm_rank*10 );
	    TEST_EQUALITY( rp1.weight(), 1.1 );
	}

	// End communication.
	communicator.end();
	TEST_ASSERT( !communicator.receiveStatus() );
    }

    // Barrier before exiting to make sure memory deallocation happened
    // correctly. 
    comm->barrier();
}

//---------------------------------------------------------------------------//
// end tstTpetraDomainCommunicator.cpp
//---------------------------------------------------------------------------//

