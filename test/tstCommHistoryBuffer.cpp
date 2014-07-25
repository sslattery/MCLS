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
 * \file   tstHistoryBuffer.cpp
 * \author Stuart Slattery
 * \brief  HistoryBuffer class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <stack>
#include <sstream>
#include <stdexcept>
#include <random>

#include <MCLS_config.hpp>
#include <MCLS_AdjointHistory.hpp>
#include <MCLS_HistoryBuffer.hpp>
#include <MCLS_CommHistoryBuffer.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_OrdinalTraits.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )	                      \
    TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( type, name, int, int )      \
    TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( type, name, int, long )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( type, name, int, double )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( type, name, long, int )     \
    TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( type, name, long, long )    \
    TEUCHOS_UNIT_TEST_TEMPLATE_2_INSTANT( type, name, long, double )

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

// Get the default communicator.
template<class Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal> > getDefaultComm()
{
#ifdef HAVE_MPI
    return Teuchos::DefaultComm<Ordinal>::getComm();
#else
    return Teuchos::rcp(new Teuchos::SerialComm<Ordinal>() );
#endif
}

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( CommHistoryBuffer, ping_pong, Ordinal, Scalar )
{
    typedef MCLS::AdjointHistory<Ordinal,std::mt19937> HT;
    typedef MCLS::HistoryBuffer<HT> HistoryBuffer;
    typedef MCLS::SendHistoryBuffer<HT> SendBuffer;
    typedef MCLS::ReceiveHistoryBuffer<HT> ReceiveBuffer;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<const Teuchos::Comm<int> > comm_dup = comm->duplicate();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    int num_histories = 5;
    HT::setByteSize();
    HistoryBuffer::setSizePackedHistory( HT::getPackedBytes() );
    HistoryBuffer::setMaxNumHistories( num_histories );

    std::size_t byte_size = num_histories * HT::getPackedBytes()
			    + sizeof(int);

    std::stack<Teuchos::RCP<HT> > bank;

    // At least 2 processes required.
    if ( comm_size > 1 )
    {
	if ( comm_rank == 0 )
	{
	    SendBuffer buffer( comm, comm_dup );
	    buffer.allocate();
	    TEST_EQUALITY( buffer.allocatedSize(), byte_size );
	    TEST_ASSERT( !buffer.status() );

	    HT h1( 1, 1 );
	    HT h2( 2, 2 );
	    HT h3( 3, 3 );
	    buffer.bufferHistory( h1 );
	    buffer.bufferHistory( h2 );
	    buffer.bufferHistory( h3 );
	    TEST_EQUALITY( buffer.numHistories(), 3 );

	    buffer.send( 1 );

	    TEST_EQUALITY( buffer.numHistories(), 0 );
	    TEST_EQUALITY( buffer.allocatedSize(), byte_size );
	    TEST_ASSERT( buffer.isEmpty() );
	    TEST_ASSERT( !buffer.status() );

	    ReceiveBuffer receiver( comm, comm_dup );
	    receiver.allocate();
	    TEST_ASSERT( receiver.isEmpty() );

	    receiver.receive( 1 );
	    TEST_ASSERT( receiver.isEmpty() );
	}

	if ( comm_rank == 1 )
	{
	    ReceiveBuffer buffer( comm, comm_dup );
	    buffer.allocate();
	    TEST_EQUALITY( buffer.allocatedSize(), byte_size );
	    TEST_ASSERT( !buffer.status() );
	    TEST_ASSERT( buffer.isEmpty() );

	    buffer.receive( 0 );
	    TEST_EQUALITY( buffer.numHistories(), 3 );
	    TEST_ASSERT( !buffer.isEmpty() );
	    buffer.addToBank( bank );
	    TEST_ASSERT( buffer.isEmpty() );
	    TEST_EQUALITY( buffer.allocatedSize(), byte_size );

	    Teuchos::RCP<HT> ph1, ph2, ph3;

	    TEST_EQUALITY( bank.size(), 3 );
	    ph3 = bank.top();
	    bank.pop();
	    TEST_EQUALITY( ph3->state(), 3 );
	    TEST_EQUALITY( ph3->weight(), 3 );

	    TEST_EQUALITY( bank.size(), 2 );
	    ph2 = bank.top();
	    bank.pop();
	    TEST_EQUALITY( ph2->state(), 2 );
	    TEST_EQUALITY( ph2->weight(), 2 );

	    TEST_EQUALITY( bank.size(), 1 );
	    ph1 = bank.top();
	    bank.pop();
	    TEST_EQUALITY( ph1->state(), 1 );
	    TEST_EQUALITY( ph1->weight(), 1 );

	    TEST_ASSERT( bank.empty() );

	    SendBuffer sender( comm, comm_dup );
	    sender.allocate();
	    TEST_EQUALITY( sender.numHistories(), 0 );
	    TEST_ASSERT( sender.isEmpty() );

	    sender.send( 0 );
	    sender.allocate();
	    TEST_EQUALITY( sender.numHistories(), 0 );
	    TEST_ASSERT( sender.isEmpty() );
	}
    }
}

UNIT_TEST_INSTANTIATION( CommHistoryBuffer, ping_pong )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( CommHistoryBuffer, non_blocking, Ordinal, Scalar )
{
    typedef MCLS::AdjointHistory<Ordinal,std::mt19937> HT;
    typedef MCLS::SendHistoryBuffer<HT> SendBuffer;
    typedef MCLS::ReceiveHistoryBuffer<HT> ReceiveBuffer;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<const Teuchos::Comm<int> > comm_dup = comm->duplicate();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    // 4 processes only.
    if ( comm_size == 4 )
    {
	HT::setByteSize();
	int num_histories = 3;

	// Comm Patterns:
	//    0 -> 1,2
	//    1 -> 0,3
	//    2 -> 0,3
	//    3 -> 1,2
	Teuchos::Array<ReceiveBuffer> 
	    receives( 2, ReceiveBuffer( comm, comm_dup,
					HT::getPackedBytes(), num_histories ) );

	Teuchos::Array<SendBuffer> 
	    sends( 2, SendBuffer( comm, comm_dup,
				  HT::getPackedBytes(), num_histories ) );

	 // post the receives
	 if ( comm_rank == 0 )
	 {
	     receives[0].post(1);
	     receives[1].post(2);
	 }
	 else if ( comm_rank == 1 )
	 {
	     receives[0].post(0);
	     receives[1].post(3);
	 }
	 else if ( comm_rank == 2 )
	 {
	     receives[0].post(0);
	     receives[1].post(3);
	 }
	 else if ( comm_rank == 3 )
	 {
	     receives[0].post(1);
	     receives[1].post(2);
	 }

	 HT h1( 1, 1 );
	 HT h2( 2, 2 );
	 HT h3( 3, 3 );
	 HT h4( 4, 4 );
	 sends[0].bufferHistory( h1 );
	 sends[1].bufferHistory( h2 );
	 sends[1].bufferHistory( h3 );
	 sends[1].bufferHistory( h4 );
	 TEST_EQUALITY( sends[0].numHistories(), 1 );
	 TEST_EQUALITY( sends[1].numHistories(), 3 );

	 TEST_ASSERT( !receives[0].check() );
	 TEST_ASSERT( !receives[1].check() );

	 TEST_ASSERT( receives[0].status() );
	 TEST_ASSERT( receives[1].status() );

	 TEST_ASSERT( !sends[0].isEmpty() );
	 TEST_ASSERT( !sends[1].isEmpty() );

	 comm->barrier();

	 if ( comm_rank == 0 )
	 {
	     sends[0].post(1);
	     sends[1].post(2);
	 }
	 else if ( comm_rank == 1 )
	 {
	     sends[0].post(0);
	     sends[1].post(3);
	 }
	 else if ( comm_rank == 2 )
	 {
	     sends[0].post(0);
	     sends[1].post(3);
	 }
	 else if ( comm_rank == 3 )
	 {
	     sends[0].post(1);
	     sends[1].post(2);
	 }

	 TEST_ASSERT( sends[0].status() );
	 TEST_ASSERT( sends[1].status() );

	 sends[0].wait();
	 sends[1].wait();

	 comm->barrier();

	 TEST_ASSERT( !sends[0].status() );
	 TEST_ASSERT( !sends[1].status() );
	 TEST_ASSERT( sends[0].isEmpty() );
	 TEST_ASSERT( sends[1].isEmpty() );

	 receives[0].wait();
	 while( !receives[1].check() );

	 TEST_ASSERT( !receives[0].status() );
	 TEST_ASSERT( !receives[1].status() );

	 std::stack<Teuchos::RCP<HT> > bank;

	 if ( comm_rank == 0 )
	 {
	     TEST_EQUALITY( receives[0].numHistories(), 1 );
	     TEST_EQUALITY( receives[1].numHistories(), 1 );

	     receives[0].addToBank( bank );
	     receives[1].addToBank( bank );
	     TEST_EQUALITY( bank.size(), 2 );

	     Teuchos::RCP<HT> hp2, hp1;
	     hp2 = bank.top();
	     bank.pop();
	     hp1 = bank.top();
	     bank.pop();

	     TEST_EQUALITY( hp2->state(), 1 );
	     TEST_EQUALITY( hp2->weight(), 1 );

	     TEST_EQUALITY( hp1->state(), 1 );
	     TEST_EQUALITY( hp1->weight(), 1 );
	 } 
	 else if ( comm_rank == 1 )
	 {
	     TEST_EQUALITY( receives[0].numHistories(), 1 );
	     TEST_EQUALITY( receives[1].numHistories(), 1 );

	     receives[0].addToBank( bank );
	     receives[1].addToBank( bank );
	     TEST_EQUALITY( bank.size(), 2 );

	     Teuchos::RCP<HT> hp2, hp1;
	     hp2 = bank.top();
	     bank.pop();
	     hp1 = bank.top();
	     bank.pop();

	     TEST_EQUALITY( hp2->state(), 1 );
	     TEST_EQUALITY( hp2->weight(), 1 );

	     TEST_EQUALITY( hp1->state(), 1 );
	     TEST_EQUALITY( hp1->weight(), 1 );
	 }
	 else if ( comm_rank == 2 )
	 {
	     TEST_EQUALITY( receives[0].numHistories(), 3 );
	     TEST_EQUALITY( receives[1].numHistories(), 3 );

	     receives[0].addToBank( bank );
	     receives[1].addToBank( bank );
	     TEST_EQUALITY( bank.size(), 6 );

	     Teuchos::RCP<HT> hp6, hp5, hp4, hp3, hp2, hp1;
	     hp6 = bank.top();
	     bank.pop();
	     hp5 = bank.top();
	     bank.pop();
	     hp4 = bank.top();
	     bank.pop();
	     hp3 = bank.top();
	     bank.pop();
	     hp2 = bank.top();
	     bank.pop();
	     hp1 = bank.top();
	     bank.pop();

	     TEST_EQUALITY( hp6->state(), 4);
	     TEST_EQUALITY( hp6->weight(), 4 );

	     TEST_EQUALITY( hp5->state(), 3 );
	     TEST_EQUALITY( hp5->weight(), 3 );

	     TEST_EQUALITY( hp4->state(), 2 );
	     TEST_EQUALITY( hp4->weight(), 2 );

	     TEST_EQUALITY( hp3->state(), 4 );
	     TEST_EQUALITY( hp3->weight(), 4 );

	     TEST_EQUALITY( hp2->state(), 3 );
	     TEST_EQUALITY( hp2->weight(), 3 );

	     TEST_EQUALITY( hp1->state(), 2 );
	     TEST_EQUALITY( hp1->weight(), 2 );
	 }
	 else if ( comm_rank == 3 )
	 {
	     TEST_EQUALITY( receives[0].numHistories(), 3 );
	     TEST_EQUALITY( receives[1].numHistories(), 3 );

	     receives[0].addToBank( bank );
	     receives[1].addToBank( bank );
	     TEST_EQUALITY( bank.size(), 6 );

	     Teuchos::RCP<HT> hp6, hp5, hp4, hp3, hp2, hp1;
	     hp6 = bank.top();
	     bank.pop();
	     hp5 = bank.top();
	     bank.pop();
	     hp4 = bank.top();
	     bank.pop();
	     hp3 = bank.top();
	     bank.pop();
	     hp2 = bank.top();
	     bank.pop();
	     hp1 = bank.top();
	     bank.pop();

	     TEST_EQUALITY( hp6->state(), 4 );
	     TEST_EQUALITY( hp6->weight(), 4 );

	     TEST_EQUALITY( hp5->state(), 3 );
	     TEST_EQUALITY( hp5->weight(), 3 );

	     TEST_EQUALITY( hp4->state(), 2 );
	     TEST_EQUALITY( hp4->weight(), 2 );

	     TEST_EQUALITY( hp3->state(), 4 );
	     TEST_EQUALITY( hp3->weight(), 4 );

	     TEST_EQUALITY( hp2->state(), 3 );
	     TEST_EQUALITY( hp2->weight(), 3 );

	     TEST_EQUALITY( hp1->state(), 2 );
	     TEST_EQUALITY( hp1->weight(), 2 );
	 }
	 comm->barrier();
     }
}

UNIT_TEST_INSTANTIATION( CommHistoryBuffer, non_blocking )

//---------------------------------------------------------------------------//
// end tstCommHistoryBuffer.cpp
//---------------------------------------------------------------------------//
