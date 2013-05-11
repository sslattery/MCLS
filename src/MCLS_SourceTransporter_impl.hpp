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
 * \file MCLS_SourceTransporter_impl.hpp
 * \author Stuart R. Slattery
 * \brief SourceTransporter class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOURCETRANSPORTER_IMPL_HPP
#define MCLS_SOURCETRANSPORTER_IMPL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_CommTools.hpp"
#include "MCLS_GlobalRNG.hpp"
#include "MCLS_Events.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Source>
SourceTransporter<Source>::SourceTransporter( 
    const Teuchos::RCP<const Comm>& comm,
    const Teuchos::RCP<Domain>& domain, 
    const Teuchos::ParameterList& plist )
    : d_comm( comm )
    , d_domain( domain )
    , d_domain_transporter( d_domain, plist )
    , d_domain_communicator( d_domain, d_comm, plist )
    , d_num_done_handles( d_comm->getSize() - 1 )
    , d_num_done_report( d_comm->getSize() - 1 )
    , d_complete_report( Teuchos::rcp(new int(0)) )
    , d_num_done( Teuchos::rcp(new int(0)) )
    , d_complete( Teuchos::rcp(new int(0)) )
{
    MCLS_REQUIRE( !d_comm.is_null() );
    MCLS_REQUIRE( !d_domain.is_null() );

    // Set the duplicate communicators. This is how we get around not having
    // access to message tags through the abstract Teuchos::Comm interface. We
    // are constructing a separate messaging space for each of these
    // bookeeping operations.
    d_comm_num_done = d_comm->duplicate();
    d_comm_complete = d_comm->duplicate();

    // Create the history count reports.
    Teuchos::Array<Teuchos::RCP<int> >::iterator report_it;
    for ( report_it = d_num_done_report.begin();
	  report_it != d_num_done_report.end();
	  ++report_it )
    {
	*report_it = Teuchos::rcp( new int(0) );
    }

    // Set the check frequency. For every d_check_freq histories run, we will
    // check for incoming histories. Default to 1.
    d_check_freq = 1;
    if ( plist.isParameter("MC Check Frequency") )
    {
	d_check_freq = plist.get<int>("MC Check Frequency");
    }

    MCLS_ENSURE( d_check_freq > 0 );
    MCLS_ENSURE( !d_comm_num_done.is_null() );
    MCLS_ENSURE( !d_comm_complete.is_null() );
}

//---------------------------------------------------------------------------//
/*!
* \brief Assign the source.
*/
template<class Source>
void SourceTransporter<Source>::assignSource(
    const Teuchos::RCP<Source>& source,
    const double relative_weight_cutoff )
{
    MCLS_REQUIRE( !source.is_null() );
    d_source = source;

    d_domain_transporter.setCutoff( relative_weight_cutoff );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport the source histories and all subsequent histories through
 * the domain to completion.
 */
template<class Source>
void SourceTransporter<Source>::transport()
{
    MCLS_REQUIRE( !d_source.is_null() );

    // Barrier before transport.
    d_comm->barrier();

    // Initialize.
    *d_complete = 0;
    *d_num_done = 0;
    d_num_done_local = 0;
    d_num_src = 0;
    d_num_run = 0;

    // Get the number of histories in the set from the source.
    d_nh = ST::numToTransportInSet( *d_source );

    // Create a history bank.
    BankType bank;
    MCLS_CHECK( bank.empty() );

    // Everyone posts receives for history buffers to get started.
    d_domain_communicator.post();

    // Post asynchronous communcations with MASTER for bookeeping.
    postMasterCount();

    // Transport all histories through the global domain until completion.
    while ( !(*d_complete) )
    {
	// Transport the source histories.
	if ( !ST::empty(*d_source) )
	{
	    transportSourceHistory( bank );
	}

	// If the source is empty, transport the rest of the histories in the
	// bank and histories that have been received from other domains.
	else if ( !bank.empty() )
	{
	    transportBankHistory( bank );
	}

	// If we're out of source and bank histories, send all buffers that
	// aren't empty.
	else if ( d_domain_communicator.send() > 0 )
	{
	    continue;
	}

	// See if we've received any histories.
	else if ( d_domain_communicator.checkAndPost(bank) > 0 )
	{
	    continue;
	}

	// If everything looks like it is finished locally, report to MASTER
	// to check if transport is done.
	else
	{
	    updateMasterCount();
	}
    }

    // Barrier before continuing.
    d_comm->barrier();

    // End all communication.
    MCLS_CHECK( !d_domain_communicator.sendBufferSize() );
    MCLS_CHECK( bank.empty() );
    d_domain_communicator.end();
    MCLS_CHECK( !d_domain_communicator.sendStatus() );
    MCLS_CHECK( !d_domain_communicator.receiveStatus() );

    // Barrier before completion.
    d_comm->barrier();

    MCLS_ENSURE( ST::empty(*d_source) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a source history.
 */
template<class Source>
void SourceTransporter<Source>::transportSourceHistory( BankType& bank )
{
    MCLS_REQUIRE( !d_source.is_null() );
    MCLS_REQUIRE( !ST::empty(*d_source) );

    // Get a history from the source.
    Teuchos::RCP<HistoryType> history = ST::getHistory( *d_source );
    MCLS_CHECK( !history.is_null() );
    MCLS_CHECK( history->alive() );

    // Add to the source history count.
    ++d_num_src;

    // Transport the history through the local domain and communicate it if
    // needed. 
    localHistoryTransport( history, bank );

    // Check for incoming histories on the check frequency. Transport those
    // that we do get.
    if ( d_num_run % d_check_freq == 0 )
    {
	if ( d_domain_communicator.checkAndPost(bank) )
	{
	    while ( !bank.empty() )
	    {
		transportBankHistory( bank );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a bank history.
 */
template<class Source>
void SourceTransporter<Source>::transportBankHistory( BankType& bank )
{
    MCLS_REQUIRE( !bank.empty() );

    // Get a history from the bank.
    Teuchos::RCP<HistoryType> history = bank.top();
    bank.pop();
    MCLS_CHECK( !history.is_null() );

    // If the history doesn't have a random number state, supply it with the
    // global RNG.
    if ( !history->rng().assigned() )
    {
	MCLS_CHECK( GlobalRNG::d_rng.assigned() );
	history->setRNG( GlobalRNG::d_rng );
    }

    // Set the history alive for transport.
    history->live();

    // Transport the history through the local domain and communicate it if
    // needed. 
    localHistoryTransport( history, bank );

    // Check for incoming histories. Do not transport these.
    if ( d_num_run % d_check_freq == 0 )
    {
	d_domain_communicator.checkAndPost(bank);
    }    
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a history through the local domain.
 */
template<class Source>
void SourceTransporter<Source>::localHistoryTransport( 
    const Teuchos::RCP<HistoryType>& history, 
    BankType& bank )
{
    MCLS_REQUIRE( !history.is_null() );
    MCLS_REQUIRE( history->alive() );
    MCLS_REQUIRE( history->rng().assigned() );

    // Do local transport.
    d_domain_transporter.transport( *history );
    MCLS_CHECK( !history->alive() );

    // Update the run count.
    ++d_num_run;

    // Communicate the history if it left the local domain.
    if ( Event::BOUNDARY == history->event() )
    {
	d_domain_communicator.communicate( history );
    }

    // Otherwise the history was killed by the weight cutoff.
    else
    {
	MCLS_CHECK( Event::CUTOFF == history->event() );
	++(*d_num_done);
	++d_num_done_local;
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Post communications with the set master proc for end of cycle.
 */
template<class Source>
void SourceTransporter<Source>::postMasterCount()
{
    // MASTER will receive history count data from each worker node.
    if ( d_comm->getRank() == MASTER )
    {
	MCLS_CHECK( d_num_done_handles.size() == d_comm->getSize() - 1 );
	MCLS_CHECK( d_num_done_report.size() == d_comm->getSize() - 1 );

	// Post an asynchronous receive for each worker.
	for ( int n = 1; n < d_comm->getSize(); ++n )
	{
	    *(d_num_done_report[n-1]) = 0;
	    d_num_done_handles[n-1] = Teuchos::ireceive<int,int>( 
		*d_comm_num_done, d_num_done_report[n-1], n );
	}
    }

    // The worker nodes post a receive from the master about completion.
    else
    {
	d_complete_handle = Teuchos::ireceive<int,int>( 
	    *d_comm_complete, d_complete_report, MASTER );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Complete communications with the set master proc for end of cycle by
 * completing all outstanding requests.
 */
template<class Source>
void SourceTransporter<Source>::completeMasterCount()
{
    // MASTER will wait for each worker node to report their completed number
    // of histories.
    Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr;
    if ( d_comm->getRank() == MASTER )
    {
	for ( int n = 1; n < d_comm->getSize(); ++n )
	{
	    request_ptr = 
		Teuchos::Ptr<Teuchos::RCP<Request> >(&d_num_done_handles[n-1]);
	    Teuchos::wait( *d_comm_num_done, request_ptr );
	    MCLS_CHECK( d_num_done_handles[n-1].is_null() );
	}
    }

    // Worker nodes send the finish message to the master.
    else
    {
	Teuchos::RCP<int> clear = Teuchos::rcp( new int(1) );
	Teuchos::RCP<Request> finish = Teuchos::isend<int,int>(
	    *d_comm_num_done, clear, MASTER );

	request_ptr = 
	    Teuchos::Ptr<Teuchos::RCP<Request> >(&finish);
	Teuchos::wait( *d_comm_num_done, request_ptr );
	MCLS_CHECK( finish.is_null() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the master count of completed histories.
 */
template<class Source>
void SourceTransporter<Source>::updateMasterCount()
{
    // MASTER checks for received reports of updated counts from work nodes
    // and adds them to the running total.
    if ( d_comm->getRank() == MASTER )
    {
	// Check if we are done with transport.
	if ( *d_num_done == d_nh ) *d_complete = 1;

	// Check on work node reports.
	int n = 0;
	while( !(*d_complete) && ++n < d_comm->getSize() )
	{
	    // Receive completed reports and repost.
	    if ( CommTools::isRequestComplete(d_num_done_handles[n-1]) )
	    {
		MCLS_CHECK( *(d_num_done_report[n-1]) > 0 );
		d_num_done_handles[n-1] = Teuchos::null;

		// Add to the running total.
		*d_num_done += *(d_num_done_report[n-1]);
		MCLS_CHECK( *d_num_done <= d_nh );

		// Repost.
		d_num_done_handles[n-1] = Teuchos::ireceive<int,int>(
		    *d_comm_num_done, d_num_done_report[n-1], n );

		// See if we are done.
		if ( *d_num_done == d_nh ) *d_complete = 1;
	    }
	}
	
	// If we finished after this update, send out the completion message.
	if ( *d_complete )
	{
	    for ( int n = 1; n < d_comm->getSize(); ++n )
	    {
		Teuchos::RCP<Request> complete = Teuchos::isend<int,int>(
		    *d_comm_complete, d_complete, n );
		Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr(&complete);
		Teuchos::wait( *d_comm_complete, request_ptr );
		MCLS_CHECK( complete.is_null() );
	    }
	}
    }

    // Worker nodes send their completed totals to the MASTER and then check
    // to see if a message has arrived indicating that transport has been
    // completed. 
    else
    {
	// Only report if we've done work.
	if ( *d_num_done > 0 )
	{
	    Teuchos::RCP<Request> report = Teuchos::isend<int,int>(
		*d_comm_num_done, d_num_done, MASTER );
	    Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr(&report);
	    Teuchos::wait( *d_comm_num_done, request_ptr );
	    MCLS_CHECK( report.is_null() );

	    *d_num_done = 0;
	}

	// Check for completion status from master.
	if ( CommTools::isRequestComplete(d_complete_handle) )
	{
	    MCLS_CHECK( *d_complete_report ==  1 );
	    d_complete_handle = Teuchos::null;
	    *d_complete = 1;
	}
    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SOURCETRANSPORTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SourceTransporter_impl.hpp
//---------------------------------------------------------------------------//

