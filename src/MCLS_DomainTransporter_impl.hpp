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
 * \file MCLS_DomainTransporter_impl.hpp
 * \author Stuart R. Slattery
 * \brief DomainTransporter implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_DOMAINTRANSPORTER_IMPL_HPP
#define MCLS_DOMAINTRANSPORTER_IMPL_HPP

#include <limits>

#include "MCLS_DBC.hpp"
#include "MCLS_Events.hpp"

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
DomainTransporter<Domain>::DomainTransporter( 
    const Teuchos::RCP<Domain>& domain )
    : d_domain( domain )
    , d_tally( DT::domainTally(*d_domain) )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );
    MCLS_REQUIRE( Teuchos::nonnull(d_tally) );
}

//---------------------------------------------------------------------------//
/*
 * \brief Transport a history through the domain.
 */
template<class Domain>
void DomainTransporter<Domain>::transport( HistoryType& history )
{
    MCLS_REQUIRE( HT::alive(history) );
    MCLS_CHECK( DT::isGlobalState(*d_domain, HT::globalState(history)) );

    // Set the history to transition.
    HT::setEvent( history, Event::TRANSITION );

    // Set the local state of the history.
    DT::setHistoryLocalState( *d_domain, history );

    // While the history is alive inside of this domain, transport it. If the
    // history leaves this domain, it is not alive with respect to this
    // domain. 
    while ( HT::alive(history) )
    {
	MCLS_CHECK( Event::TRANSITION == HT::event(history) );
	MCLS_CHECK( !DT::terminateHistory(*d_domain,history) );
	MCLS_CHECK( HT::weightAbs(history) < std::numeric_limits<double>::max() );
	MCLS_CHECK( DT::isGlobalState(*d_domain, HT::globalState(history)) );

	// Tally the history.
	TT::tallyHistory( *d_tally, history );

	// Transition the history one step.
	DT::processTransition( *d_domain, history );

	// See if we should kill the history because it has met the
	// termination condition for the domain. If so, kill it and post
	// process. The history is complete.
	if ( DT::terminateHistory(*d_domain,history) )
	{
	    HT::setEvent( history, Event::CUTOFF );
	    HT::kill( history );
	    TT::postProcessHistory( *d_tally, history );
	}

	// If the history has left the domain, kill it. The history will
	// continue in a different domain.
	else if ( DT::isBoundaryState(*d_domain,HT::globalState(history)) )
	{
            HT::setEvent( history, Event::BOUNDARY );
            HT::kill( history );
	}
    }

    MCLS_ENSURE( !HT::alive(history) );
    MCLS_ENSURE( Event::TRANSITION != HT::event(history) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_DOMAINTRANSPORTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_DomainTransporter_impl.hpp
//---------------------------------------------------------------------------//

