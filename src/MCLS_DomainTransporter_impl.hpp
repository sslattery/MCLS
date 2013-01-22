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

#include <MCLS_DBC.hpp>
#include <MCLS_Events.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
DomainTransporter<Domain>::DomainTransporter( 
    const Teuchos::RCP<Domain>& domain, const Teuchos::ParameterList& plist )
    : d_domain( domain )
    , d_tally( d_domain->domainTally() )
{
    Require( !d_domain.is_null() );
    Require( !d_tally.is_null() );

    d_weight_cutoff = plist.get<double>("Relative Weight Cutoff");
    Ensure( d_weight_cutoff > 0.0 );
}

//---------------------------------------------------------------------------//
/*
 * \brief Transport a history through the domain.
 */
template<class Domain>
void DomainTransporter<Domain>::transport( HistoryType& history )
{
    Require( history.alive() );
    Require( history.rng().assigned() );
    Require( history.weightAbs() >= d_weight_cutoff );
    Require( d_domain->isLocalState(history.state()) );

    // Set the history to transition.
    history.setEvent( TRANSITION );

    // While the history is alive inside of this domain, transport it. If the
    // history leaves this domain, it is not alive with respect to this
    // domain. 
    while ( history.alive() )
    {
	Check( history.event() == TRANSITION );
	Check( history.weightAbs() >= d_weight_cutoff );
	Check( d_domain->isLocalState(history.state()) );

	// Tally the history.
	d_tally->tallyHistory( history );

	// Transition the history one step.
	d_domain->processTransition( history );

	// If the history's weight is less than the cutoff, kill it.
	if ( history.weightAbs() < d_weight_cutoff )
	{
	    history.setEvent( CUTOFF );
	    history.kill();
	}

	// If the history has left the domain, kill it.
	else if ( !d_domain->isLocalState(history.state()) )
	{
	    history.setEvent( BOUNDARY );
	    history.kill();
	}
    }

    Ensure( !history.alive() );
    Ensure( history.event() != TRANSITION );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS


#endif // end MCLS_DOMAINTRANSPORTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_DomainTransporter_impl.hpp
// ---------------------------------------------------------------------------//
