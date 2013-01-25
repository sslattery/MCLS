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
