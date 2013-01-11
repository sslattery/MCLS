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
TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( HistoryBuffer, sizes, Ordinal, Scalar )
{
    typedef MCLS::History<Scalar,Ordinal> HT;

    MCLS::HistoryBuffer<HT> buffer_1;
    TEST_EQUALITY( buffer_1.allocatedSize(), 0 );
    TEST_ASSERT( buffer_1.isEmpty() );
    TEST_ASSERT( !buffer_1.isFull() );

    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::maxNum(), 1000 );
    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::sizePackedHistory(), 0 );

    HT::setByteSize( 0 );
    MCLS::HistoryBuffer<HT>::setSizePackedHistory( HT::getPackedBytes() );
    MCLS::HistoryBuffer<HT>::setMaxNumHistory( 10 );
    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::maxNum(), 10 );
    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::sizePackedHistory(),
		  sizeof(Scalar) + sizeof(Ordinal) );

    MCLS::HistoryBuffer<HT> buffer_2;
    TEST_EQUALITY( buffer_2.allocatedSize(), 0 );
    TEST_ASSERT( buffer_2.isEmpty() );
    TEST_ASSERT( !buffer_2.isFull() );

    buffer_2.allocate();

    TEST_EQUALITY( buffer_2.allocatedSize(),
		   10 * (sizeof(Scalar) + sizeof(Ordinal)) + sizeof(int) );
    TEST_ASSERT( buffer_2.isEmpty() );
    TEST_ASSERT( !buffer_2.isFull() );

    buffer_2.deallocate();
    TEST_EQUALITY( buffer_2.allocatedSize(), 0 );
    TEST_ASSERT( buffer_2.isEmpty() );
    TEST_ASSERT( !buffer_2.isFull() );

    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::maxNum(), 10 );
    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::sizePackedHistory(),
		   sizeof(Scalar) + sizeof(Ordinal) );
}

UNIT_TEST_INSTANTIATION( HistoryBuffer, sizes )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_2_DECL( HistoryBuffer, buffering, Ordinal, Scalar )
{
    typedef MCLS::History<Scalar,Ordinal> HT;
    HT::setByteSize( 0 );

    int num_history = 4;
    MCLS::HistoryBuffer<HT> buffer( HT::getPackedBytes(), num_history );
    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::maxNum(), 4 );
    TEST_EQUALITY( MCLS::HistoryBuffer<HT>::sizePackedHistory(),
		   sizeof(Scalar) + sizeof(Ordinal) );

    TEST_EQUALITY( buffer.allocatedSize(),
		   num_history*(sizeof(Scalar)+sizeof(Ordinal)) + sizeof(int) );
    TEST_ASSERT( buffer.isEmpty() );
    TEST_EQUALITY( buffer.numHistories(), 0 );

    std::stack<Teuchos::RCP<HT> > stack;
    TEST_ASSERT( stack.empty() );

    HT h1( 1, 1 );
    HT h2( 2, 2 );
    HT h3( 3, 3 );
    HT h4( 4, 4 );

    buffer.bufferHistory( h1 );
    TEST_ASSERT( !buffer.isFull() );
    TEST_ASSERT( !buffer.isEmpty() );
    TEST_EQUALITY( buffer.numHistories(), 1 );

    buffer.bufferHistory( h2 );
    TEST_ASSERT( !buffer.isFull() );
    TEST_ASSERT( !buffer.isEmpty() );
    TEST_EQUALITY( buffer.numHistories(), 2 );

    buffer.bufferHistory( h3 );
    TEST_ASSERT( !buffer.isFull() );
    TEST_ASSERT( !buffer.isEmpty() );
    TEST_EQUALITY( buffer.numHistories(), 3 );

    buffer.bufferHistory( h4 );
    TEST_ASSERT( buffer.isFull() );
    TEST_ASSERT( !buffer.isEmpty() );
    TEST_EQUALITY( buffer.numHistories(), 4 );

    buffer.addToStack( stack );

    Teuchos::RCP<HT> ph1, ph2, ph3, ph4;

    TEST_EQUALITY( stack.size(), 4 );
    ph4 = stack.top();
    stack.pop();
    TEST_EQUALITY( ph4->state(), 4 );
    TEST_EQUALITY( ph4->weight(), 4 );

    TEST_EQUALITY( stack.size(), 3 );
    ph3 = stack.top();
    stack.pop();
    TEST_EQUALITY( ph3->state(), 3 );
    TEST_EQUALITY( ph3->weight(), 3 );

    TEST_EQUALITY( stack.size(), 2 );
    ph2 = stack.top();
    stack.pop();
    TEST_EQUALITY( ph2->state(), 2 );
    TEST_EQUALITY( ph2->weight(), 2 );

    TEST_EQUALITY( stack.size(), 1 );
    ph1 = stack.top();
    stack.pop();
    TEST_EQUALITY( ph1->state(), 1 );
    TEST_EQUALITY( ph1->weight(), 1 );

    TEST_ASSERT( stack.empty() );
}

UNIT_TEST_INSTANTIATION( HistoryBuffer, buffering )

//---------------------------------------------------------------------------//
// end HistoryBuffer.cpp
//---------------------------------------------------------------------------//
