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
 * \file   tstForwardHistory.cpp
 * \author Stuart Slattery
 * \brief  ForwardHistory class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <random>

#include <MCLS_config.hpp>
#include <MCLS_ForwardHistory.hpp>
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
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( ForwardHistory, history, Ordinal )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = getDefaultComm<int>();

    MCLS::ForwardHistory<Ordinal> h_1;
    TEST_EQUALITY( h_1.weight(), Teuchos::ScalarTraits<double>::one() );
    TEST_EQUALITY( h_1.globalState(), Teuchos::OrdinalTraits<Ordinal>::invalid() );
    TEST_EQUALITY( h_1.startingState(), 
		   Teuchos::OrdinalTraits<Ordinal>::zero() );
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

    h_1.setGlobalState( 3 );
    TEST_EQUALITY( h_1.globalState(), 3 );

    h_1.setLocalState( 4 );
    TEST_EQUALITY( h_1.localState(), 4 );

    h_1.setStartingState( 2 );
    TEST_EQUALITY( h_1.startingState(), 2 );

    h_1.setWeight( 5 );
    TEST_EQUALITY( h_1.weight(), 5 );

    h_1.addWeight( 2 );
    TEST_EQUALITY( h_1.weight(), 7 );

    h_1.multiplyWeight( -2 );
    TEST_EQUALITY( h_1.weight(), -14 );
    TEST_EQUALITY( h_1.weightAbs(), 14 );

    TEST_EQUALITY( h_1.historyTally(), 0.0 );
    h_1.addToHistoryTally( 1.34 );
    TEST_EQUALITY( h_1.historyTally(), 1.34 );

    MCLS::ForwardHistory<Ordinal> h_2( 5, 2, 6 );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.globalState(), 5 );
    TEST_EQUALITY( h_2.localState(), 2 );
    TEST_EQUALITY( h_2.startingState(), 5 );
    TEST_ASSERT( !h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::Event::NO_EVENT );
    TEST_EQUALITY( h_2.historyTally(), 0.0 );
}

UNIT_TEST_INSTANTIATION( ForwardHistory, history )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( ForwardHistory, pack_unpack, Ordinal )
{
    std::size_t byte_size = 2*sizeof(Ordinal) + 2*sizeof(double) + 2*sizeof(int);
    MCLS::ForwardHistory<Ordinal>::setByteSize();
    std::size_t packed_bytes =
	MCLS::ForwardHistory<Ordinal>::getPackedBytes();
    TEST_EQUALITY( packed_bytes, byte_size );

    MCLS::ForwardHistory<Ordinal> h_1( 5, 2, 6 );
    h_1.live();
    h_1.setEvent( MCLS::Event::BOUNDARY );
    h_1.addToHistoryTally( 2.44 );
    Teuchos::Array<char> packed_history = h_1.pack();
    TEST_EQUALITY( Teuchos::as<std::size_t>( packed_history.size() ), 
		   byte_size );

    MCLS::ForwardHistory<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.globalState(), 5 );
    TEST_EQUALITY( h_2.localState(), Teuchos::OrdinalTraits<Ordinal>::invalid() );
    TEST_EQUALITY( h_2.startingState(), 5 );
    TEST_ASSERT( h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::Event::BOUNDARY );
    TEST_EQUALITY( h_2.historyTally(), 2.44 );
}

UNIT_TEST_INSTANTIATION( ForwardHistory, pack_unpack )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( ForwardHistory, broadcast, Ordinal )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();

    MCLS::ForwardHistory<Ordinal>::setByteSize();
    std::size_t packed_bytes =
	MCLS::ForwardHistory<Ordinal>::getPackedBytes();
    Teuchos::Array<char> packed_history( packed_bytes );

    if ( comm_rank == 0 )
    {
	MCLS::ForwardHistory<Ordinal> h_1( 5, 2, 6 );
	h_1.live();
	h_1.setEvent( MCLS::Event::BOUNDARY );
	h_1.addToHistoryTally( 1.98 );
	packed_history = h_1.pack();
    }

    Teuchos::broadcast( *comm, 0, packed_history() );

    MCLS::ForwardHistory<Ordinal> h_2( packed_history );
    TEST_EQUALITY( h_2.weight(), 6 );
    TEST_EQUALITY( h_2.globalState(), 5 );
    TEST_EQUALITY( h_2.localState(), Teuchos::OrdinalTraits<Ordinal>::invalid() );
    TEST_EQUALITY( h_2.startingState(), 5 );
    TEST_ASSERT( h_2.alive() );
    TEST_EQUALITY( h_2.event(), MCLS::Event::BOUNDARY );
    TEST_EQUALITY( h_2.historyTally(), 1.98 );
}

UNIT_TEST_INSTANTIATION( ForwardHistory, broadcast )

//---------------------------------------------------------------------------//
// end tstForwardHistory.cpp
//---------------------------------------------------------------------------//
