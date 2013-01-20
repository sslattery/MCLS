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
    , d_sends( d_domain->numNeighbors() )
    , d_receives( d_domain->numNeighbors() )
    , d_num_neighbors( d_domain->numNeighbors() )
{
    Require( !d_domain.isNull() );
    Require( !d_comm.isNull() );
    Require( d_num_neighbors >= 0 );

    Insist( HistoryType::packedBytes(), "Packed history size not set." );
    HistoryBufferType::setSizePackedHistory( HistoryType::packedBytes() );

    // Get the max number of histories that will be stored in each buffer.
    if ( plist.isParameter("History Buffer Size") )
    {
	HistoryBufferType::setMaxNumHistories( 
	    plist.get<int>("History Buffer Size") );
    }

    // Allocate the send and receive buffers.
    for ( int n = 0; n < d_num_neighbors; ++n )
    {
	d_sends[n].allocate();
	d_receives[n].allocate();
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Buffer and send a history.
 */
template<class Domain>
const DomainCommunicator<Domain>::Result& 
DomainCommunicator<Domain>::communicate( 
    const Teuchos::RCP<HistoryType>& history )
{
    Require( !history.is_null() );

    // Initialize result status.
    d_result.sent = false;
    d_result.destination = 0;

    // Add the history to the appropriate buffer.
    int neighbor_id = d_domain->owningNeighbor( history->state() );
    d_sends[neighbor_id].bufferHistory( *history );

    // Update the result destination.
    d_result.destination = d_domain->neighborRank(neighbor_id);

    // If the buffer is full send it.
    if ( d_sends[neighbor_id].isFull() )
    {
	Check( d_sends[neighbor_id].numHistories() == maxBufferSize() );

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

}

//---------------------------------------------------------------------------//
/*!
 * \brief Flush all buffers whether they are empty or not.
 */
template<class Domain>
int DomainCommunicator<Domain>::flush()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Post receives.
 */
template<class Domain>
void DomainCommunicator<Domain>::post()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Wait on receive buffers.
 */
template<class Domain>
int DomainCommunicator<Domain>::wait( BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Receive buffers and repost.
 */
template<class Domain>
int DomainCommunicator<Domain>::checkAndPost( BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Status of send buffers.
 */
template<class Domain>
bool DomainCommunicator<Domain>::sendStatus()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Status of receive buffers.
 */
template<class Domain>
bool DomainCommunicator<Domain>::receiveStatus()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief End communication.
 */
template<class Domain>
void DomainCommunicator<Domain>::end()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Number of particles in all buffers.
 */
template<class Domain>
std::size_t DomainCommunicator<Domain>::sendBufferSize() const
{

}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_DOMAINCOMMUNICATOR_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_DomainCommunicator_impl.hpp
//---------------------------------------------------------------------------//

