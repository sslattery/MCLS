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
 * \file   tstSPRNG.cpp
 * \author Stuart Slattery
 * \brief  SPRNG class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <MCLS_config.hpp>
#include <MCLS_SPRNG.hpp>

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

int seed = 493875348;

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SPRNG, random_test )
{
    int *id1, *id2, *id3;
    int num = 5;

    id1 = init_sprng(0, num, seed, 1);
    id2 = init_sprng(1, num, seed, 1);
    id3 = init_sprng(0, num, seed, 1);
    
    MCLS::SPRNG ran1(id1, 0);	
    MCLS::SPRNG ran2(id2, 1);
    TEST_EQUALITY( ran1.getIndex(), 0 );
    TEST_EQUALITY( ran2.getIndex(), 1 );

    int num_rand = 10000;

    double r1 = 0.0, r2 = 0.0;
    for (int i = 0; i < num_rand; i++)
    {
        r1 += ran1.random();
        r2 += ran2.random();
    }
    TEST_FLOATING_EQUALITY( r1/num_rand, 0.5, 0.01 );
    TEST_FLOATING_EQUALITY( r2/num_rand, 0.5, 0.01 );

    MCLS::SPRNG ran3(id3, 0);
    
    double r3 = 0.0;
    for (int i = 0; i < num_rand; i++)
    {
        r3 += ran3.random();
    }
    TEST_FLOATING_EQUALITY( r1, r3, 1.0e-8 );
    
    double eps = 0.00001;
    for (int i = 0; i < 10; i++)
    {
	TEST_FLOATING_EQUALITY( ran1.random(), ran3.random(), eps );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SPRNG, sprng_test )
{
    int num  = 5;

    int *idr = init_sprng(0, num, seed, 1);
    int *id1 = init_sprng(0, num, seed, 1);

    MCLS::SPRNG ranr(idr, 0);
    MCLS::SPRNG ran1(id1, 0);
    MCLS::SPRNG ran2(ran1);
    MCLS::SPRNG ran3(ran1);
    MCLS::SPRNG empty;

    TEST_ASSERT(ran1.assigned());
    TEST_ASSERT(!empty.assigned());

    Teuchos::Array<double> ref(80);
    for (int i = 0; i < 80; i++)
    {
	ref[i] = ranr.random();
    }

    double eps = .00001;
    for (int i = 0; i < 20; i++)
    {
       TEST_FLOATING_EQUALITY( ran1.random(), ref[i], eps );
    }
    for (int i = 20; i < 40; i++)
    {
	TEST_FLOATING_EQUALITY( ran2.random(), ref[i], eps );
    }
    for (int i = 40; i < 60; i++)
    {
       TEST_FLOATING_EQUALITY( ran3.random(), ref[i], eps );
    }

    TEST_EQUALITY( ran1.getID(), id1 );
    TEST_EQUALITY( ran2.getID(), id1 );
    TEST_EQUALITY( ran3.getID(), id1 );

    ranr = ran2;
    for (int i = 60; i < 80; i++)
    {
	TEST_FLOATING_EQUALITY( ranr.random(), ref[i], eps );
    }

    TEST_EQUALITY( ranr.getID(), id1 );

    ran3 = empty;
    TEST_ASSERT( !ran3.assigned() );

    MCLS::SPRNG ran4(ran3);
    TEST_ASSERT( !ran4.assigned() );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SPRNG, pack_test )
{
    int num = 5;

    Teuchos::Array<double> ref(80);
    Teuchos::Array<char> buffer;

    {
	int *id1 = init_sprng(0, num, seed, 1);
	int *idr = init_sprng(0, num, seed, 1);
	
	MCLS::SPRNG ran1(id1, 0);
	MCLS::SPRNG ranr(idr, 0);

	for (int i = 0; i < 80; i++)
	{
	    ref[i] = ranr.random();
	}

	for (int i = 0; i < 40; i++)
	{
	    ran1.random();
	}

	buffer = ran1.pack();

	TEST_EQUALITY( Teuchos::as<std::size_t>( buffer.size() ),
		       ran1.getSize() );
    }

    {
	MCLS::SPRNG uran(buffer);

	double r   = 0;
	double rf  = 0;
	for (int i = 0; i < 40; i++)
	{
	    r  = uran.random();
	    rf = ref[i+40];
	    TEST_FLOATING_EQUALITY( r, rf, 1.0e-8 );
	}
    }
}

//---------------------------------------------------------------------------//
// end tstSPRNG.cpp
//---------------------------------------------------------------------------//
