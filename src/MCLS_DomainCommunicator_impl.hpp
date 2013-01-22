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
 * \file MCLS_DomainCommunicator_impl.hpp
 * \author Stuart R. Slattery
 * \brief DomainCommunicator class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_DOMAINCOMMUNICATOR_IMPL_HPP
#define MCLS_DOMAINCOMMUNICATOR_IMPL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_Events.hpp"

#include <Teuchos_as.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
//---------------------------------------------------------------------------//
template<class Domain>
DomainCommunicator<Domain>::DomainCommunicator( 
    const Teuchos::RCP<Domain>& domain,
    const Teuchos::RCP<const Comm>& set_const_comm,
    const Teuchos::ParameterList& plist )
    : d_domain( domain )
    , d_comm( set_const_comm )
    , d_sends( d_domain->numSendNeighbors() )
    , d_receives( d_domain->numReceiveNeighbors() )
    , d_num_send_neighbors( d_domain->numSendNeighbors() )
    , d_num_receive_neighbors( d_domain->numReceiveNeighbors() )
{
    Require( !d_domain.is_null() );
    Require( !d_comm.is_null() );
    Require( d_num_send_neighbors >= 0 );
    Require( d_num_receive_neighbors >= 0 );

    Insist( HistoryType::getPackedBytes(), "Packed history size not set." );
    HistoryBufferType::setSizePackedHistory( HistoryType::getPackedBytes() );

    // Get the max number of histories that will be stored in each buffer.
    if ( plist.isParameter("History Buffer Size") )
    {
	HistoryBufferType::setMaxNumHistories( 
	    plist.get<int>("History Buffer Size") );
    }

    // Allocate the send buffers and set their communicators.
    for ( int n = 0; n < d_num_send_neighbors; ++n )
    {
	d_sends[n].setComm( d_comm );
	d_sends[n].allocate();
    }

    // Allocate the receive buffers and set their communicators.
    for ( int n = 0; n < d_num_receive_neighbors; ++n )
    {
	d_receives[n].setComm( d_comm );
	d_receives[n].allocate();
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Buffer and send a history.
 */
template<class Domain>
const typename DomainCommunicator<Domain>::Result& 
DomainCommunicator<Domain>::communicate( 
    const Teuchos::RCP<HistoryType>& history )
{
    Require( !history.is_null() );
    Require( history->event() == BOUNDARY );
    Require( !history->alive() );

    // Initialize result status.
    d_result.sent = false;
    d_result.destination = 0;

    // Add the history to the appropriate buffer.
    int neighbor_id = d_domain->owningNeighbor( history->state() );
    d_sends[neighbor_id].bufferHistory( *history );

    // Update the result destination.
    d_result.destination = d_domain->sendNeighborRank( neighbor_id );
    Check( d_result.destination < d_comm->getSize() );

    // If the buffer is full send it.
    if ( d_sends[neighbor_id].isFull() )
    {
	Check( d_sends[neighbor_id].numHistories() == 
	       Teuchos::as<int>(maxBufferSize()) );

	d_sends[neighbor_id].post( d_result.destination );
	d_sends[neighbor_id].wait();

	Check( d_sends[neighbor_id].isEmpty() );
	Check( d_sends[neighbor_id].allocatedSize() > 0 );
	Check( !d_sends[neighbor_id].status() );

	d_result.sent = true;
    }

    // Return the result.
    return d_result;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Send all buffers that are not empty.
 */
template<class Domain>
int DomainCommunicator<Domain>::send()
{
    int num_sent = 0;

    for ( int n = 0; n < d_num_send_neighbors; ++n )
    {
	Check( d_sends[n].allocatedSize() > 0 );
	Check( d_domain->sendNeighborRank(n) < d_comm->getSize() );

	if( !d_sends[n].isEmpty() )
	{
	    Check( d_sends[n].numHistories() > 0 );

	    num_sent += d_sends[n].numHistories();
	    d_sends[n].post( d_domain->sendNeighborRank(n) );
	    d_sends[n].wait();

	    Check( num_sent > 0 );
	}

	Ensure( d_sends[n].isEmpty() );
	Ensure( d_sends[n].allocatedSize() > 0 );
	Ensure( !d_sends[n].status() );
    }

    return num_sent;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Send all buffers whether they are empty or not.
 */
template<class Domain>
int DomainCommunicator<Domain>::flush()
{
    int num_sent = 0;

    for ( int n = 0; n < d_num_send_neighbors; ++n )
    {
	Check( d_sends[n].allocatedSize() > 0 );
	Check( d_domain->sendNeighborRank(n) < d_comm->getSize() );

	num_sent += d_sends[n].numHistories();
	d_sends[n].post( d_domain->sendNeighborRank(n) );
	d_sends[n].wait();

	Ensure( d_sends[n].isEmpty() );
	Ensure( d_sends[n].allocatedSize() > 0 );
	Ensure( !d_sends[n].status() );
    }

    return num_sent;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Post receives.
 */
template<class Domain>
void DomainCommunicator<Domain>::post()
{
    for ( int n = 0; n < d_num_receive_neighbors; ++n )
    {
	Check( !d_receives[n].status() );
	Check( d_receives[n].allocatedSize() > 0 );
	Check( d_receives[n].isEmpty() );
	Check( d_domain->receiveNeighborRank(n) < d_comm->getSize() );

	d_receives[n].post( d_domain->receiveNeighborRank(n) );

	Ensure( d_receives[n].status() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Wait on receive buffers.
 */
template<class Domain>
int DomainCommunicator<Domain>::wait( BankType& bank )
{
    int num_received = 0;

    for ( int n = 0; n < d_num_receive_neighbors; ++n )
    {
	Check( d_receives[n].allocatedSize() > 0 );

	d_receives[n].wait();
	num_received += d_receives[n].numHistories();
	d_receives[n].addToBank( bank );

	Ensure( !d_receives[n].status() );
	Ensure( d_receives[n].isEmpty() );
    }

    return num_received;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Receive buffers and repost.
 */
template<class Domain>
int DomainCommunicator<Domain>::checkAndPost( BankType& bank )
{
    int num_received = 0;

    for ( int n = 0; n < d_num_receive_neighbors; ++n )
    {
	Check( d_receives[n].allocatedSize() > 0 );
	Check( d_domain->receiveNeighborRank(n) < d_comm->getSize() );

	if( d_receives[n].check() )
	{
	    num_received += d_receives[n].numHistories();
	    d_receives[n].addToBank( bank );

	    Check( d_receives[n].isEmpty() );
	    Check( !d_receives[n].status() );

	    d_receives[n].post( d_domain->receiveNeighborRank(n) );

	    Ensure( d_receives[n].status() );
	}
    }

    return num_received;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Status of send buffers.
 *
 * Return true only if all send buffers are on.
 */
template<class Domain>
bool DomainCommunicator<Domain>::sendStatus()
{
    if ( d_num_send_neighbors == 0 ) return false;
      
    for ( int n = 0; n < d_num_send_neighbors; ++n )
    {
	Check( d_sends[n].allocatedSize() > 0 );

	if ( !d_sends[n].status() ) 
	{
	    return false;
	}
    }

    return true;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Status of receive buffers.
 *
 * Return true only if all receive buffers are on.
 */
template<class Domain>
bool DomainCommunicator<Domain>::receiveStatus()
{
    if ( d_num_receive_neighbors == 0 ) return false;

    for ( int n = 0; n < d_num_receive_neighbors; ++n )
    {
	Check( d_receives[n].allocatedSize() > 0 );

	if ( !d_receives[n].status() ) 
	{
	    return false;
	}
    }

    return true;
}

//---------------------------------------------------------------------------//
/*!
 * \brief End communication. Send all buffers and receive them.
 *
 * This will clear all communication requests. All buffers will be emptied and
 * therefore all data they contain lost.
 *
 * All receives must be posted or the flush will hang.
 */
template<class Domain>
void DomainCommunicator<Domain>::end()
{
    flush();

    for ( int n = 0; n < d_num_receive_neighbors; ++n )
    {
	Check( d_receives[n].allocatedSize() > 0 );

	d_receives[n].wait();
	d_receives[n].empty();

	Ensure( d_receives[n].isEmpty() );
    }

    Ensure( !sendStatus() );
    Ensure( !receiveStatus() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Number of particles in all buffers.
 */
template<class Domain>
std::size_t DomainCommunicator<Domain>::sendBufferSize() const
{
    int send_num = 0;

    for ( int n = 0; n < d_num_send_neighbors; ++n )
    {
	Check( d_sends[n].allocatedSize() > 0 );

	send_num += d_sends[n].numHistories();
    }

    return send_num;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_DOMAINCOMMUNICATOR_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_DomainCommunicator_impl.hpp
//---------------------------------------------------------------------------//

