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
#include "MCLS_Events.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_OrdinalTraits.hpp>

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
    , d_parent( Teuchos::OrdinalTraits<int>::invalid() )
    , d_children( Teuchos::OrdinalTraits<int>::invalid(),
                  Teuchos::OrdinalTraits<int>::invalid() )
    , d_domain( domain )
    , d_domain_transporter( d_domain, plist )
    , d_domain_communicator( d_domain, d_comm, plist )
    , d_num_done_report( Teuchos::ArrayRCP<int>(1,0), Teuchos::ArrayRCP<int>(1,0) )
    , d_complete_report( Teuchos::ArrayRCP<int>(1,0) )
    , d_num_done( Teuchos::ArrayRCP<int>(1,0) )
    , d_complete( Teuchos::ArrayRCP<int>(1,0) )
    , d_num_done_tag( 19873 )
    , d_completion_status_tag( 19874 )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );

    // Get the comm parameters.
    int my_rank = d_comm->getRank();
    int my_size = d_comm->getSize();

    // Get the parent process. MASTER has no parent.
    if ( my_rank != MASTER )
    {
	if ( my_rank % 2 == 0 )
	{
	    d_parent = ( my_rank / 2 ) - 1;
	}
	else
	{
	    d_parent = ( my_rank - 1 ) / 2;
	}
    }
	 
    // Get the first child process.
    int child_1 = ( my_rank * 2 ) + 1;
    if ( child_1 < my_size )
    {
	d_children.first = child_1;
    }

    // Get the second child process.
    int child_2 = child_1 + 1;
    if ( child_2 < my_size )
    {
	d_children.second = child_2;
    }

    // Set the check frequency. For every d_check_freq histories run, we will
    // check for incoming histories. Default to 1.
    d_check_freq = 1;
    if ( plist.isParameter("MC Check Frequency") )
    {
	d_check_freq = plist.get<int>("MC Check Frequency");
    }
    
    MCLS_ENSURE( d_check_freq > 0 );
    MCLS_ENSURE( Teuchos::nonnull(d_comm) );
    MCLS_ENSURE( Teuchos::nonnull(d_comm) );
}

//---------------------------------------------------------------------------//
/*!
* \brief Assign the source.
*/
template<class Source>
void SourceTransporter<Source>::assignSource(
    const Teuchos::RCP<Source>& source )
{
    MCLS_REQUIRE( Teuchos::nonnull(source) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );
    d_source = source;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport the source histories and all subsequent histories through
 * the domain to completion.
 */
template<class Source>
void SourceTransporter<Source>::transport()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_source) );

    // Barrier before transport.
    d_comm->barrier();

    // Initialize.
    d_complete[0] = 0;
    d_num_done[0] = 0;
    d_num_run = 0;

    // Get the number of histories in the set from the source.
    d_nh = ST::numToTransportInSet( *d_source );

    // Create a history bank.
    BankType bank;
    MCLS_CHECK( bank.empty() );

    // Everyone posts receives for history buffers to get started.
    d_domain_communicator.post();

    // Post asynchronous communcations in the binary tree for history counts.
    postTreeCount();

    // Transport all histories through the global domain until completion.
    while ( !d_complete[0] )
    {
	// Transport the source histories.
	if ( !ST::empty(*d_source) )
	{
	    transportSourceHistory( bank );
	    ++d_num_run;
	}

	// If the source is empty, transport the bank histories.
	else if ( !bank.empty() )
	{
	    transportBankHistory( bank );
	    ++d_num_run;
	}

	// If we're out of source and bank histories or have hit the check
	// frequency, process incoming messages.
	if ( (ST::empty(*d_source) && bank.empty()) ||  
             d_num_run == d_check_freq )
	{
            processMessages( bank );
	    d_num_run = 0;
	}

	// If everything looks like it is finished locally, report through
        // the tree to check if transport is done.
        if ( ST::empty(*d_source) && bank.empty() )
	{
	    controlTermination();
	}
    }

    // Barrier before continuing.
    d_comm->barrier();

    // Complete the binary tree outstanding communication.
    completeTreeCount();

    // End all communication and free all buffers.
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
 * \brief Reset the state of the transporter.
 */
template<class Source>
void SourceTransporter<Source>::reset()
{
    d_num_done_report = std::make_pair( Teuchos::ArrayRCP<int>(1,0),
					Teuchos::ArrayRCP<int>(1,0) );
    d_complete_report = Teuchos::ArrayRCP<int>(1,0);
    d_num_done = Teuchos::ArrayRCP<int>(1,0);
    d_complete = Teuchos::ArrayRCP<int>(1,0);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a source history.
 */
template<class Source>
void SourceTransporter<Source>::transportSourceHistory( BankType& bank )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_source) );
    MCLS_REQUIRE( !ST::empty(*d_source) );

    // Get a history from the source.
    Teuchos::RCP<HistoryType> history = ST::getHistory( *d_source );
    MCLS_CHECK( Teuchos::nonnull(history) );
    MCLS_CHECK( HT::alive(*history) );

    // Transport the history through the local domain and communicate it if
    // needed. 
    localHistoryTransport( history, bank );
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
    MCLS_CHECK( Teuchos::nonnull(history) );

    // Set the history alive for transport.
    HT::live( *history );

    // Transport the history through the local domain and communicate it if
    // needed. 
    localHistoryTransport( history, bank );
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
    MCLS_REQUIRE( Teuchos::nonnull(history) );
    MCLS_REQUIRE( HT::alive(*history) );

    // Do local transport.
    d_domain_transporter.transport( *history );
    MCLS_CHECK( !HT::alive(*history) );

    // Communicate the history if it left the local domain.
    if ( Event::BOUNDARY == HT::event(*history) )
    {
	d_domain_communicator.communicate( history );
    }

    // Otherwise the history was killed by the finishing all of its steps.
    else
    {
	MCLS_CHECK( Event::CUTOFF == HT::event(*history) );
	++d_num_done[0];
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Process incoming messages.
 */
template<class Source>
void SourceTransporter<Source>::processMessages( BankType& bank )
{
    // Check for incoming histories.
    d_domain_communicator.checkAndPost(bank);

    // Add to the history completed tally 
    updateTreeCount();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Post communications in the binary tree.
 */
template<class Source>
void SourceTransporter<Source>::postTreeCount()
{
    // Post a receive from the first child for history count data.
    if ( d_children.first != Teuchos::OrdinalTraits<int>::invalid() )
    {
	d_num_done_report.first[0] = 0;
	d_num_done_handles.first = Teuchos::ireceive<int,int>(
	    d_num_done_report.first, d_children.first, 
	    d_num_done_tag, *d_comm );
    }

    // Post a receive from the second child for history count data.
    if ( d_children.second != Teuchos::OrdinalTraits<int>::invalid() )
    {
	d_num_done_report.second[0] = 0;
	d_num_done_handles.second = Teuchos::ireceive<int,int>(
	    d_num_done_report.second, d_children.second, 
	    d_num_done_tag, *d_comm );
    }

    // Post a receive from parent for transport completion.
    if ( d_parent != Teuchos::OrdinalTraits<int>::invalid() )
    {
	d_complete_handle = Teuchos::ireceive<int,int>( 
	    d_complete_report, d_parent, d_completion_status_tag, *d_comm );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Complete outstanding communications in the binary tree at the end of
 * a cycle.
 */
template<class Source>
void SourceTransporter<Source>::completeTreeCount()
{
    Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr;

    // Children nodes send the finish message to the parent.
    if ( d_parent != Teuchos::OrdinalTraits<int>::invalid() )
    {
	Teuchos::ArrayRCP<int> clear( 1, 1 );
	Teuchos::RCP<Request> finish = Teuchos::isend<int,int>(
	    clear, d_parent, d_num_done_tag, *d_comm );
	request_ptr = 
	    Teuchos::Ptr<Teuchos::RCP<Request> >(&finish);
	Teuchos::wait( *d_comm, request_ptr );
	MCLS_CHECK( Teuchos::is_null(finish) );
    }

    // Parent will wait for first child node to clear communication.
    if ( d_children.first != Teuchos::OrdinalTraits<int>::invalid() )
    {
        request_ptr = 
            Teuchos::Ptr<Teuchos::RCP<Request> >(&d_num_done_handles.first);
        Teuchos::wait( *d_comm, request_ptr );
        MCLS_CHECK( Teuchos::is_null(d_num_done_handles.first) );
    }

    // Parent will wait for second child node to clear communication.
    if ( d_children.second != Teuchos::OrdinalTraits<int>::invalid() )
    {
        request_ptr = 
            Teuchos::Ptr<Teuchos::RCP<Request> >(&d_num_done_handles.second);
        Teuchos::wait( *d_comm, request_ptr );
        MCLS_CHECK( Teuchos::is_null(d_num_done_handles.second) );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the binary tree count of completed histories.
 */
template<class Source>
void SourceTransporter<Source>::updateTreeCount()
{
    // Check for received reports of updated counts from first child.
    if ( d_children.first != Teuchos::OrdinalTraits<int>::invalid() )
    {
        // Receive completed reports and repost.
        if ( CommTools::isRequestComplete(d_num_done_handles.first) )
        {
            MCLS_CHECK( *(d_num_done_report.first) > 0 );
            d_num_done_handles.first = Teuchos::null;

            // Add to the running total.
            d_num_done[0] += d_num_done_report.first[0];
            MCLS_CHECK( d_num_done[0] <= d_nh );

            // Repost.
            d_num_done_handles.first = Teuchos::ireceive<int,int>(
		d_num_done_report.first, d_children.first,
		d_num_done_tag, *d_comm );
        }
    }

    // Check for received reports of updated counts from second child.
    if ( d_children.second != Teuchos::OrdinalTraits<int>::invalid() )
    {
        // Receive completed reports and repost.
        if ( CommTools::isRequestComplete(d_num_done_handles.second) )
        {
            MCLS_CHECK( d_num_done_report.second[0] > 0 );
            d_num_done_handles.second = Teuchos::null;

            // Add to the running total.
            d_num_done[0] += d_num_done_report.second[0];
            MCLS_CHECK( d_num_done[0] <= d_nh );

            // Repost.
            d_num_done_handles.second = Teuchos::ireceive<int,int>(
		d_num_done_report.second, d_children.second,
		d_num_done_tag, *d_comm );
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Send the global finished message to the children.
 */
template<class Source>
void SourceTransporter<Source>::sendCompleteToChildren()
{
    // Child 1
    if ( d_children.first != Teuchos::OrdinalTraits<int>::invalid() )
    {
        Teuchos::RCP<Request> complete = Teuchos::isend<int,int>(
	    d_complete, d_children.first, d_completion_status_tag, *d_comm );
        Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr(&complete);
        Teuchos::wait( *d_comm, request_ptr );
        MCLS_CHECK( Teuchos::is_null(complete) );
    }

    // Child 2
    if ( d_children.second != Teuchos::OrdinalTraits<int>::invalid() )
    {
        Teuchos::RCP<Request> complete = Teuchos::isend<int,int>(
	    d_complete, d_children.second, d_completion_status_tag, *d_comm );
        Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr(&complete);
        Teuchos::wait( *d_comm, request_ptr );
        MCLS_CHECK( Teuchos::is_null(complete) );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Control the termination of a stage.
 */
template<class Source>
void SourceTransporter<Source>::controlTermination()
{
    // Send any partially full buffers.
    d_domain_communicator.send();

    // Update the history count from the children.
    updateTreeCount();

    // MASTER checks for completion.
    if ( d_comm->getRank() == MASTER ) 
    {
        if ( d_num_done[0] == d_nh )
        {
            d_complete[0] = 1;
            sendCompleteToChildren();
        }
    }

    // Other nodes send the number of histories completed to parent and check
    // to see if a message has arrived from the parent indicating that
    // transport has been completed.
    else
    {
        // Send completed number of histories to parent.
        if ( d_num_done[0] > 0 )
        {
            Teuchos::RCP<Request> report = Teuchos::isend<int,int>(
		d_num_done, d_parent, d_num_done_tag, *d_comm );
            Teuchos::Ptr<Teuchos::RCP<Request> > request_ptr(&report);
            Teuchos::wait( *d_comm, request_ptr );
            MCLS_CHECK( Teuchos::is_null(report) );
            d_num_done[0] = 0;
        } 

        // Check for completion status from parent.
        if ( CommTools::isRequestComplete(d_complete_handle) )
        {
            MCLS_CHECK( d_complete_report[0] ==  1 );
            d_complete_handle = Teuchos::null;
            d_complete[0] = 1;
            sendCompleteToChildren();
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

