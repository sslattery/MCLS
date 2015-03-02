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
 * \file MCLS_UniformForwardSource_impl.hpp
 * \author Stuart R. Slattery
 * \brief UniformForwardSource class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_UNIFORMFORWARDSOURCE_IMPL_HPP
#define MCLS_UNIFORMFORWARDSOURCE_IMPL_HPP

#include <cmath>

#include "MCLS_DBC.hpp"
#include "MCLS_SamplingTools.hpp"
#include "MCLS_Serializer.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_ArrayRCP.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
UniformForwardSource<Domain>::UniformForwardSource( 
    const Teuchos::RCP<VectorType>& b,
    const Teuchos::RCP<Domain>& domain,
    const Teuchos::ParameterList& plist )
    : d_b( b )
    , d_domain( domain )
    , d_rng_dist( RDT::create(0.0, 1.0) )
    , d_nh_per_state( 1 )
    , d_current_local_state( 0 )
    , d_current_state_samples( 0 )
    , d_nh_requested( VT::getGlobalLength(*d_b) )
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight( 1.0 )
    , d_nh_left(0)
    , d_nh_emitted(0)
{
    MCLS_REQUIRE( Teuchos::nonnull(d_b) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );

    // Get the requested number of histories. The default value is a sample
    // ratio of 1. The sample ratio effectively defines the number of samples
    // per system state for a forward source.
    if ( plist.isParameter("Sample Ratio") )
    {
	d_nh_per_state = std::ceil( plist.get<double>("Sample Ratio") );
	d_nh_requested *= d_nh_per_state;
    }
    
    // Set the total to the requested amount. This may change based on the
    // global stratified sampling.
    d_nh_total = d_nh_requested;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the source.
 */
template<class Domain>
void UniformForwardSource<Domain>::buildSource()
{
    // Set the current local state.
    d_current_local_state = 0;
    d_current_state_samples = 0;
    
    // Set counters.
    d_nh_domain = VT::getLocalLength( *d_b ) * d_nh_per_state;
    d_nh_left = d_nh_domain;
    d_nh_emitted = 0;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get a history from the source.
 */
template<class Domain>
Teuchos::RCP<typename UniformForwardSource<Domain>::HistoryType> 
UniformForwardSource<Domain>::getHistory()
{
    MCLS_REQUIRE( 1.0 == d_weight );
    MCLS_REQUIRE( d_nh_left >= 0 );
    MCLS_REQUIRE( Teuchos::nonnull(d_rng) );
    MCLS_REQUIRE( d_current_state_samples < d_nh_per_state );
    MCLS_REQUIRE( d_current_local_state < VT::getLocalLength(*d_b) );

    // Return null if empty.
    if ( !d_nh_left )
    {
	return Teuchos::null;
    }

    // Generate the history.
    Teuchos::RCP<HistoryType> history = Teuchos::rcp( new HistoryType() );

    // Get a starting state.
    MCLS_CHECK( VT::isLocalRow(*d_b,d_current_local_state) );
    Ordinal starting_state = VT::getGlobalRow( *d_b, d_current_local_state );
    MCLS_CHECK( DT::isGlobalState(*d_domain,starting_state) );

    // Set the history state.
    history->setWeight( d_weight );
    history->setGlobalState( starting_state );
    history->setStartingState( starting_state );
    history->live();

    // Update local state.
    ++d_current_state_samples;
    if ( d_current_state_samples == d_nh_per_state )
    {
	++d_current_local_state;
	d_current_state_samples = 0;
    }
    
    // Update count.
    --d_nh_left;
    ++d_nh_emitted;

    MCLS_ENSURE( Teuchos::nonnull(history) );
    MCLS_ENSURE( history->alive() );
    MCLS_ENSURE( history->weightAbs() == d_weight );

    return history;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMFORWARDSOURCE_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformForwardSource_impl.hpp
//---------------------------------------------------------------------------//

