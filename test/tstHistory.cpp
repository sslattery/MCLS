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
 * \file   tstHistory.cpp
 * \author Stuart Slattery
 * \brief  History class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <MCLS_config.hpp>
#include <MCLS_RNGControl.hpp>
#include <MCLS_History.hpp>
#include <MCLS_Events.hpp>

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
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( type, name, int )           \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( type, name, long )

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

int seed = 2394723;

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( History, history, Ordinal )
{
    MCLS::RNGControl control( seed );

    MCLS::History<Ordinal> h_1;
    TEST_EQUALITY( h_1.weight(), Teuchos::ScalarTraits<double>::one() );
    TEST_EQUALITY( h_1.state(), Teuchos::OrdinalTraits<Ordinal>::zero() );
    TEST_ASSERT( !h_1.alive() );
    TEST_EQUALITY( h_1.event(), MCLS::NO_EVENT );

    h_1.live();
    TEST_ASSERT( h_1.alive() );

    h_1.kill();
    TEST_ASSERT( !h_1.alive() );

    h_1.setEvent( MCLS::CUTOFF );
    TEST_EQUALITY( h_1.event(), Teuchos::as<int>(MCLS::CUTOFF) );

    h_1.setEvent( MCLS::BOUNDARY );
    TEST_EQUALITY( h_1.event(), Teuchos::as<int>(MCLS::BOUNDARY) );

    h_1.setState( 3 );
    TEST_EQUALITY( h_1.state(), 3 );

    h_1.setWeight( 5 );
    TEST_EQUALITY( h_1.weight(), 5 );

    h_1.addWeight( 2 );
    TEST_EQUALITY( h_1.weight(), 7 );

    h_1.multiplyWeight( -2 );
    TEST_EQUALITY( h_1.weight(), -14 );
    TEST_EQUALITY( h_1.weightAbs(), 14 );

    MCLS::RNGControl::RNG rng = control.rng( 4 );
    h_1.setRNG( rng );
    TEST_EQUALITY( h_1.rng().getIndex(), 4 );

    MCLS::History<Ordinal> h_2( 5, 6 );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( !h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::NO_EVENT );
}

UNIT_TEST_INSTANTIATION( History, history )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( History, pack_unpack, Ordinal )
{
    MCLS::RNGControl control( seed );
    MCLS::RNGControl::RNG ranr = control.rng( 4 );
    Teuchos::Array<double> ref(80);
    for (int i = 0; i < 80; i++)
    {
	ref[i] = ranr.random();
    }

    MCLS::RNGControl::RNG rng = control.rng( 4 );
    for (int i = 0; i < 40; i++)
    {
	rng.random();
    }

    std::size_t byte_size = 
	control.getSize() + sizeof(Ordinal) + sizeof(double) + 2*sizeof(int);
    MCLS::History<Ordinal>::setByteSize( control.getSize() );
    std::size_t packed_bytes =
	MCLS::History<Ordinal>::getPackedBytes();
    TEST_EQUALITY( packed_bytes, byte_size );

    MCLS::History<Ordinal> h_1( 5, 6 );
    h_1.setRNG( rng );
    h_1.live();
    h_1.setEvent( MCLS::BOUNDARY );
    Teuchos::Array<char> packed_history = h_1.pack();
    TEST_EQUALITY( Teuchos::as<std::size_t>( packed_history.size() ), 
		   byte_size );

    MCLS::History<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::BOUNDARY );

    MCLS::RNGControl::RNG rng_2 = h_2.rng();
    for (int i = 0; i < 40; i++)
    {
	TEST_FLOATING_EQUALITY( rng_2.random(), ref[i+40], 1.0e-8 );
    }
}

UNIT_TEST_INSTANTIATION( History, pack_unpack )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( History, pack_unpack_no_rng, Ordinal )
{
    std::size_t byte_size = sizeof(Ordinal) + sizeof(double) + 2*sizeof(int);
    MCLS::History<Ordinal>::setByteSize( 0 );

    MCLS::History<Ordinal> h_1( 5, 6 );
    h_1.live();
    Teuchos::Array<char> packed_history = h_1.pack();
    TEST_EQUALITY( Teuchos::as<std::size_t>( packed_history.size() ), 
		   byte_size );

    MCLS::History<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( h_2.alive() );
}

UNIT_TEST_INSTANTIATION( History, pack_unpack_no_rng )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( History, broadcast, Ordinal )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();

    MCLS::RNGControl control( seed );
    MCLS::RNGControl::RNG ranr = control.rng( 4 );
    Teuchos::Array<double> ref(80);
    for (int i = 0; i < 80; i++)
    {
	ref[i] = ranr.random();
    }

    MCLS::History<Ordinal>::setByteSize( control.getSize() );
    std::size_t packed_bytes =
	MCLS::History<Ordinal>::getPackedBytes();
    Teuchos::Array<char> packed_history( packed_bytes );

    if ( comm_rank == 0 )
    {
	MCLS::RNGControl::RNG rng = control.rng( 4 );
	for (int i = 0; i < 40; i++)
	{
	    rng.random();
	}

	MCLS::History<Ordinal> h_1( 5, 6 );
	h_1.setRNG( rng );
	h_1.live();
	h_1.setEvent( MCLS::BOUNDARY );
	packed_history = h_1.pack();
    }

    Teuchos::broadcast( *comm, 0, packed_history() );

    MCLS::History<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::BOUNDARY );

    MCLS::RNGControl::RNG rng_2 = h_2.rng();
    for (int i = 0; i < 40; i++)
    {
	TEST_FLOATING_EQUALITY( rng_2.random(), ref[i+40], 1.0e-8 );
    }
}

UNIT_TEST_INSTANTIATION( History, broadcast )

//---------------------------------------------------------------------------//
// end tstHistory.cpp
//---------------------------------------------------------------------------//
