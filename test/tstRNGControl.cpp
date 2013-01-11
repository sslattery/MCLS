//---------------------------------------------------------------------------//
/*!
 * \file   tstRNGControl.cpp
 * \author Stuart Slattery
 * \brief  RNGControl class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <MCLS_config.hpp>
#include <MCLS_RNGControl.hpp>

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_Array.hpp"
#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_as.hpp"

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
// Random number seed.
//---------------------------------------------------------------------------//

int seed = 2452423;

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SPRNG, control_test )
{
    typedef MCLS::RNGControl::RNG RNG;

    MCLS::RNGControl control( seed );

    TEST_EQUALITY( control.getNumber(), 1000000000 );
    TEST_EQUALITY( control.getSeed(), 2452423 );
    TEST_EQUALITY( control.getIndex(), 0 );

    RNG r0  = control.rng();
    TEST_EQUALITY( control.getIndex(), 1 );
    RNG r1  = control.rng();
    TEST_EQUALITY( control.getIndex(), 2 );
    RNG r2  = control.rng();
    TEST_EQUALITY( control.getIndex(), 3 );

    RNG rr2 = control.rng(2);
    TEST_EQUALITY( control.getIndex(), 3 );

    RNG rr1 = control.rng(1);
    TEST_EQUALITY( control.getIndex(), 2 );

    control.setIndex(0);
    RNG rr0 = control.rng();
    TEST_EQUALITY( control.getIndex(), 1 );

    for (int i = 0; i < 100; i++)
    {
	double rn0  = r0.random();
	double rrn0 = rr0.random();
	double rn1  = r1.random();
	double rrn1 = rr1.random();
	double rn2  = r2.random();
	double rrn2 = rr2.random();

	TEST_FLOATING_EQUALITY( rn0, rrn0, 1.0e-6 );
	TEST_FLOATING_EQUALITY( rn1, rrn1, 1.0e-6 );
	TEST_FLOATING_EQUALITY( rn2, rrn2, 1.0e-6 );

	TEST_INEQUALITY( rn0, rrn1 );
	TEST_INEQUALITY( rn1, rrn2 );
	TEST_INEQUALITY( rn2, rrn0 );
    }

    Teuchos::Array<char> r0_packed = r0.pack();
    TEST_EQUALITY( control.getSize(), 
		   Teuchos::as<std::size_t>( r0_packed.size() ) );
}

//---------------------------------------------------------------------------//
// end tstRNGControl.cpp
//---------------------------------------------------------------------------//
