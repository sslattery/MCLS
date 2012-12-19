//---------------------------------------------------------------------------//
/*!
 * \file   tstAssertion.cpp
 * \author Stuart Slattery
 * \brief  Assertion class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <MCLS_config.hpp>
#include <MCLS_Assertion.hpp>

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_CommHelpers.hpp"

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
// Check that a MCLS::Assertion looks different than a
// std::runtime_error as it inherits from std::logic_error.
TEUCHOS_UNIT_TEST( Assertion, differentiation_test )
{
    try
    {
	throw std::runtime_error( "runtime error" );
    }
    catch( const MCLS::Assertion& assertion )
    {
	TEST_ASSERT( 0 );
    }
    catch( ... )
    {
	TEST_ASSERT( 1 );
    }
}

//---------------------------------------------------------------------------//
// Check that a MCLS::Assertion can be caught and the appropriate
// error message is written.
TEUCHOS_UNIT_TEST( Assertion, message_test )
{
    std::string message;

    try
    {
	throw MCLS::Assertion( "cond", "file", 12 );
    }
    catch( const MCLS::Assertion& assertion )
    {
	message = std::string( assertion.what() );
    }
    catch( ... )
    {
	TEST_ASSERT( 0 );
    }

    const std::string true_message( 
	"MCLS Assertion: cond, failed in file, line 12.\n" );
    TEST_ASSERT( 0 == message.compare( true_message ) );
}

//---------------------------------------------------------------------------//
// Check that we can throw a nemesis assertion with throwAssertion.
TEUCHOS_UNIT_TEST( Assertion, throw_test )
{
    try
    {
	const std::string message( "message" );
	const std::string file( "file" );
	const int line( 12 );
	MCLS::throwAssertion( message, file, line );
	throw std::runtime_error( "this shouldn't be thrown" );
    }    
    catch( const MCLS::Assertion& assertion )
    {
	TEST_ASSERT( 1 );	
    }
    catch( ... )
    {
	TEST_ASSERT( 0 );
    }
}

//---------------------------------------------------------------------------//
// Test the precondition check for DBC.
TEUCHOS_UNIT_TEST( Assertion, precondition_test )
{
    try 
    {
	testPrecondition( 0 );
	throw std::runtime_error( "this shouldn't be thrown" );
    }
    catch( const MCLS::Assertion& assertion )
    {
#if HAVE_CHIMERA_DBC
	std::string message( assertion.what() );
	std::string true_message( "MCLS Assertion: 0, failed in" );
	std::string::size_type idx = message.find( true_message );
	if ( idx == std::string::npos )
	{
	    TEST_ASSERT( 0 );
	}
#else
	TEST_ASSERT( 0 );
#endif
    }
    catch( ... )
    {
#if HAVE_CHIMERA_DBC
	TEST_ASSERT( 0 );
#endif
    }
}

//---------------------------------------------------------------------------//
// Test the postcondition check for DBC.
TEUCHOS_UNIT_TEST( Assertion, postcondition_test )
{
    try 
    {
	testPostcondition( 0 );
	throw std::runtime_error( "this shouldn't be thrown" );
    }
    catch( const MCLS::Assertion& assertion )
    {
#if HAVE_CHIMERA_DBC
	std::string message( assertion.what() );
	std::string true_message( "MCLS Assertion: 0, failed in" );
	std::string::size_type idx = message.find( true_message );
	if ( idx == std::string::npos )
	{
	    TEST_ASSERT( 0 );
	}
#else
	TEST_ASSERT( 0 );
#endif
    }
    catch( ... )
    {
#if HAVE_CHIMERA_DBC
	TEST_ASSERT( 0 );
#endif
    }
}

//---------------------------------------------------------------------------//
// Test the invariant check for DBC.
TEUCHOS_UNIT_TEST( Assertion, invariant_test )
{
    try 
    {
	testInvariant( 0 );
	throw std::runtime_error( "this shouldn't be thrown" );
    }
    catch( const MCLS::Assertion& assertion )
    {
#if HAVE_CHIMERA_DBC
	std::string message( assertion.what() );
	std::string true_message( "MCLS Assertion: 0, failed in" );
	std::string::size_type idx = message.find( true_message );
	if ( idx == std::string::npos )
	{
	    TEST_ASSERT( 0 );
	}
#else
	TEST_ASSERT( 0 );
#endif
    }
    catch( ... )
    {
#if HAVE_CHIMERA_DBC
	TEST_ASSERT( 0 );
#endif
    }
}

//---------------------------------------------------------------------------//
// Test that we can remember a value and check it with DBC.
TEUCHOS_UNIT_TEST( Assertion, remember_test )
{
    remember( int test_value_1 = 0 );
    remember( int test_value_2 = 1 );
 
    try 
    {
	testInvariant( test_value_1 );
    }
    catch( const MCLS::Assertion& assertion )
    {
#if HAVE_CHIMERA_DBC
	TEST_ASSERT( 1 );
#else
	TEST_ASSERT( 0 );
#endif
    }
    catch( ... )
    {
#if HAVE_CHIMERA_DBC
	TEST_ASSERT( 0 );
#endif
    }

    try 
    {
	testInvariant( test_value_2 );
	TEST_ASSERT( 1 );
    }
    catch( ... )
    {
	TEST_ASSERT( 0 );
    }
}

//---------------------------------------------------------------------------//
// Test the assertion check for DBC.
TEUCHOS_UNIT_TEST( Assertion, assertion_test )
{
    try 
    {
	testAssertion( 0 );
	throw std::runtime_error( "this shouldn't be thrown" );
    }
    catch( const MCLS::Assertion& assertion )
    {
	std::string message( assertion.what() );
	std::string true_message( "MCLS Assertion: 0, failed in" );
	std::string::size_type idx = message.find( true_message );
	if ( idx == std::string::npos )
	{
	    TEST_ASSERT( 0 );
	}
    }
    catch( ... )
    {
	TEST_ASSERT( 0 );
    }
}

//---------------------------------------------------------------------------//
// end tstAssertion.cpp
//---------------------------------------------------------------------------//
