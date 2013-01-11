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

#include <MCLS_config.hpp>
#include <MCLS_History.hpp>
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
    typedef MCLS::History<Scalar,Ordinal> HT;
    typedef MCLS::HistoryBuffer<HT> HistoryBuffer;
    typedef MCLS::SendHistoryBuffer<HT> SendBuffer;
    typedef MCLS::ReceiveHistoryBuffer<HT> ReceiveBuffer;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    int num_histories = 5;
    HT::setByteSize( 0 );
    HistoryBuffer::setSizePackedHistory( HT::getPackedBytes() );
    HistoryBuffer::setMaxNumHistories( num_histories );

    std::size_t byte_size = num_histories * HT::getPackedBytes()
			    + sizeof(int);

    std::stack<Teuchos::RCP<HT> > bank;

    if ( comm_size > 1 )
    {
	if ( comm_rank == 0 )
	{
	    SendBuffer buffer( comm );
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

	    ReceiveBuffer receiver( comm );
	    receiver.allocate();
	    TEST_ASSERT( receiver.isEmpty() );

	    receiver.receive( 1 );
	    TEST_ASSERT( receiver.isEmpty() );
	}

	if ( comm_rank == 1 )
	{
	    ReceiveBuffer buffer( comm );
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

	    SendBuffer sender( comm );
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
// end tstCommHistoryBuffer.cpp
//---------------------------------------------------------------------------//
