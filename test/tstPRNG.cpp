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
 * \file   tstPRNG.cpp
 * \author Stuart Slattery
 * \brief  PRNG class unit tests.
 */
//---------------------------------------------------------------------------//

#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <MCLS_PRNG.hpp>
#include <MCLS_Xorshift.hpp>

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
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( PRNG, prng_test, RNG )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = getDefaultComm<int>();
    
    MCLS::PRNG<RNG> prng( comm->getRank() );
    
    // Make a set of random numbers on each process.
    int num_random = 1000;
    Teuchos::Array<double> rands( num_random * comm->getSize(), 0.0 );
    std::uniform_real_distribution<double> rand_dist(0.0,1.0);
    for ( int i = 0; i < num_random; ++i )
    {
	rands[comm->getRank()*num_random + i] = prng.random( rand_dist );
    }
    
    // Collect all random numbers on each process.
    Teuchos::Array<double> global_rands( num_random * comm->getSize(), 0.0 );
    Teuchos::reduceAll<int,double>( *comm, Teuchos::REDUCE_SUM, 
				    global_rands.size(),
				    rands.getRawPtr(), global_rands.getRawPtr() );

    // Check that the random numbers on each process are unique.
    for ( int i = 0; i < comm->getSize(); ++i )
    {
	for ( int j = 0; j < comm->getSize(); ++j )
	{
	    if ( i != j )
	    {
		for ( int k = 0; k < num_random; ++k )
		{
		    TEST_INEQUALITY( global_rands[i*num_random + k],
				     global_rands[j*num_random + k] );
		}
	    }
	}
    }
}

typedef std::mt19937 mt19937;
typedef std::mt19937_64 mt1993764;
typedef MCLS::Xorshift<> Xorshift;

TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( PRNG, prng_test, mt19937 )
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( PRNG, prng_test, mt1993764 )
TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( PRNG, prng_test, Xorshift )

//---------------------------------------------------------------------------//
// end tstPRNG.cpp
//---------------------------------------------------------------------------//
