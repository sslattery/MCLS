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
// end HistoryBuffer.cpp
//---------------------------------------------------------------------------//
