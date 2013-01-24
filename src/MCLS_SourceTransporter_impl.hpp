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

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
SourceTransporter<Domain>::SourceTransporter( 
    const Teuchos::RCP<const Comm>& comm,
    const Teuchos::RCP<Domain>& domain, 
    const Teuchos::ParameterList& plist )
    : d_comm( comm )
    , d_domain( domain )
    , d_tally( d_domain->domainTally() )
    , d_domain_transporter( d_domain, plist )
    , d_domain_communicator( d_domain, d_comm, plist )
    , d_num_done_handles( d_comm->getSize() - 1 )
    , d_num_done_report( d_comm->getSize() - 1, Teuchos::rcp(new int(0)) )
    , d_complete_report( Teuchos::rcp(new int(0)) )
    , d_complete( Teuchos::rcp(new int(0)) )
{
    Require( !d_comm.is_null() );
    Require( !d_domain.is_null() );
    Require( !d_tally.is_null() );

    // Set the check frequency.
    d_check_freq = 1;
    if ( plist.isParameter("MC Check Frequency") )
    {
	d_check_freq = plist.get<int>("MC Check Frequency");
    }

    Ensure( d_check_freq > 0 );
}

//---------------------------------------------------------------------------//
/*!
* \brief Assign the source.
*/
template<class Domain>
void SourceTransporter<Domain>::assignSource(
    const Teuchos::RCP<SourceType>& source )
{
    Require( !source.is_null() );
    d_source = source;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport the source histories and all subsequent histories through
 * the domain to completion.
 */
template<class Domain>
void SourceTransporter<Domain>::transport()
{
    Require( !d_source.is_null() );

    // Barrier before transport.
    d_comm->barrier();

    // Initialize.
    *d_complete = 0;
    d_num_done = 0;
    d_num_done_local = 0;
    d_num_src = 0;
    d_num_run = 0;

    // Get the number of histories in the set from the source.
    d_nh = d_source->numToTransportInSet();

    // Create a history bank.
    BankType bank;
    Check( bank.empty() );

    // Everyone posts receives to get started.
    d_domain_communicator.post();

    // Post asynchronous communcations with MASTER for bookeeping.
    postMasterCount();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a source history.
 */
template<class Domain>
void SourceTransporter<Domain>::transportSourceHistory( BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a bank history.
 */
template<class Domain>
void SourceTransporter<Domain>::transportBankHistory( BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a history through the local domain.
 */
template<class Domain>
void SourceTransporter<Domain>::localHistoryTransport( 
    const Teuchos::RCP<HistoryType>& history, 
    BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Post communications with the set master proc for end of cycle.
 */
template<class Domain>
void SourceTransporter<Domain>::postMasterCount()
{
    // MASTER will receive history count data from each worker node.
    if ( d_comm->getRank() == MASTER )
    {
	Check( d_num_done_handles.size() == d_comm->getSize() - 1 );
	Check( d_num_done_report.size() == d_comm->getSize() - 1 );

	// Post an asynchronous receive for each worker.
	for ( int n = 1; n < d_comm->getSize(); ++n )
	{
	    *d_num_done_report[n-1] = 0;
	    d_num_done_handles[n-1] = Teuchos::ireceive<int,int>( 
		*d_comm, d_num_done_report[n-1], n );
	}
    }

    // The worker nodes post a receive from the master about completion.
    else
    {
	d_complete_handle = 
	    Teuchos::ireceive<int,int>( *d_comm, d_complete_report, MASTER );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Complete communications with the set master proc for end of cycle.
 */
template<class Domain>
void SourceTransporter<Domain>::completeMasterCount()
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
	    Teuchos::wait( *d_comm, request_ptr );
	    Check( d_num_done_handles[n-1].is_null() );
	}
    }

    // Worker nodes send the finish message to the master.
    else
    {
	Teuchos::RCP<int> clear = Teuchos::rcp( new int(1) );
	Teuchos::RCP<Request> finish = Teuchos::isend<int,int>(
	    *d_comm, clear, MASTER );

	request_ptr = 
	    Teuchos::Ptr<Teuchos::RCP<Request> >(&finish);
	Teuchos::wait( *d_comm, request_ptr );
	Check( finish.is_null() );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the master count of completed histories.
 */
template<class Domain>
void SourceTransporter<Domain>::updateMasterCount()
{
    // MASTER checks for received reports of updated counts from work nodes
    // and adds them to the running total.
    if ( d_comm->getRank() == MASTER )
    {
	// Check if we are done with transport.
	if ( d_num_done == d_nh ) *d_complete = 1;

	// Check on work node reports.
	int n = 0;
	while( !(*d_complete) && ++n < d_comm->getSize() );
	{
	    // Receive completed reports and repost.

	}
	
    }

    // Worker nodes send their completed totals to the MASTER and then check
    // to see if a message has arrived indicating that transport has been
    // completed. 
    else
    {

    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SOURCETRANSPORTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SourceTransporter_impl.hpp
//---------------------------------------------------------------------------//
