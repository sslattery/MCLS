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
 * \file   tstAdjointHistory.cpp
 * \author Stuart Slattery
 * \brief  AdjointHistory class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <random>

#include <MCLS_config.hpp>
#include <MCLS_AdjointHistory.hpp>
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
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( AdjointHistory, history, Ordinal )
{
    MCLS::AdjointHistory<Ordinal> h_1;
    TEST_EQUALITY( h_1.weight(), Teuchos::ScalarTraits<double>::one() );
    TEST_EQUALITY( h_1.state(), Teuchos::OrdinalTraits<Ordinal>::zero() );
    TEST_ASSERT( !h_1.alive() );
    TEST_EQUALITY( h_1.event(), MCLS::Event::NO_EVENT );

    h_1.live();
    TEST_ASSERT( h_1.alive() );

    h_1.kill();
    TEST_ASSERT( !h_1.alive() );

    h_1.setEvent( MCLS::Event::CUTOFF );
    TEST_EQUALITY( h_1.event(), Teuchos::as<int>(MCLS::Event::CUTOFF) );

    h_1.setEvent( MCLS::Event::BOUNDARY );
    TEST_EQUALITY( h_1.event(), Teuchos::as<int>(MCLS::Event::BOUNDARY) );

    h_1.setState( 3 );
    TEST_EQUALITY( h_1.state(), 3 );

    h_1.setWeight( 5 );
    TEST_EQUALITY( h_1.weight(), 5 );

    h_1.addWeight( 2 );
    TEST_EQUALITY( h_1.weight(), 7 );

    h_1.multiplyWeight( -2 );
    TEST_EQUALITY( h_1.weight(), -14 );
    TEST_EQUALITY( h_1.weightAbs(), 14 );

    MCLS::AdjointHistory<Ordinal> h_2( 5, 6 );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( !h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::Event::NO_EVENT );
}

UNIT_TEST_INSTANTIATION( AdjointHistory, history )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( AdjointHistory, pack_unpack, Ordinal )
{
    std::size_t byte_size = sizeof(Ordinal) + sizeof(double) + 2*sizeof(int);
    MCLS::AdjointHistory<Ordinal>::setByteSize();
    std::size_t packed_bytes = 
	MCLS::AdjointHistory<Ordinal>::getPackedBytes();
    TEST_EQUALITY( packed_bytes, byte_size );

    MCLS::AdjointHistory<Ordinal> h_1( 5, 6 );
    h_1.live();
    h_1.setEvent( MCLS::Event::BOUNDARY );
    Teuchos::Array<char> packed_history = h_1.pack();
    TEST_EQUALITY( Teuchos::as<std::size_t>( packed_history.size() ), 
		   byte_size );

    MCLS::AdjointHistory<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::Event::BOUNDARY );
}

UNIT_TEST_INSTANTIATION( AdjointHistory, pack_unpack )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( AdjointHistory, broadcast, Ordinal )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();

    MCLS::AdjointHistory<Ordinal>::setByteSize();
    std::size_t packed_bytes = 
	MCLS::AdjointHistory<Ordinal>::getPackedBytes();
    Teuchos::Array<char> packed_history( packed_bytes );

    if ( comm_rank == 0 )
    {
	MCLS::AdjointHistory<Ordinal> h_1( 5, 6 );
	h_1.live();
	h_1.setEvent( MCLS::Event::BOUNDARY );
	packed_history = h_1.pack();
    }

    Teuchos::broadcast( *comm, 0, packed_history() );

    MCLS::AdjointHistory<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.state(), 5 );
    TEST_ASSERT( h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::Event::BOUNDARY );
}

UNIT_TEST_INSTANTIATION( AdjointHistory, broadcast )

//---------------------------------------------------------------------------//
// end tstAdjointHistory.cpp
//---------------------------------------------------------------------------//
