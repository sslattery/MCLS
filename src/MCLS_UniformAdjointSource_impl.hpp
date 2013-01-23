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
 * \file MCLS_UniformSource_impl.hpp
 * \author Stuart R. Slattery
 * \brief UniformSource class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_UNIFORMSOURCE_IMPL_HPP
#define MCLS_UNIFORMSOURCE_IMPL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_GlobalRNG.hpp"
#include "MCLS_SamplingTools.hpp"

#include "Teuchos_ArrayRCP.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
UniformSource<Domain>::UniformSource( 
    const Teuchos::RCP<VectorType>& b,
    const Teuchos::RCP<Domain>& domain,
    const Teuchos::RCP<RNGControl>& rng_control,
    const Teuchos::RCP<const Comm>& comm,
    Teuchos::ParameterList& plist )
    : Base( b, domain, rng_control )
    , d_comm( comm )
    , d_rng_stream(0)
    , d_nh_requested( VT::getGlobalLength(*Base::b_b) )
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight( VT::norm1(*Base::b_b) )
    , d_nh_left(0)
    , d_nh_emitted(0)
{
    Require( !d_comm.is_null() );

    // Get the requested global number of histories.
    if ( plist.isParameter("Global Number of Histories") )
    {
	d_nh_requested = plist.get<int>("Global Number of Histories");
    }
    
    // Set the total to the requested amount. This may change.
    d_nh_total = d_nh_requested;

    // Set the relative weight cutoff with the source weight.
    double cutoff = plist.get<double>( "Weight Cutoff" );
    plist.set<double>( "Relative Weight Cutoff", cutoff*d_weight );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the source.
 */
template<class Domain>
void UniformSource<Domain>::buildSource()
{
    // Set the RNG stream.
    makeRNG();

    // For now, we'll sample the same number of histories from the source in
    // each domain. Obviously, this will bias the solution and instead we will
    // have to implement a global sampling tool (perhaps global stratified and
    // locally random).
    d_nh_domain = d_nh_total / d_comm->getSize();

    // The total size may have changed due to integer rounding.
    d_nh_total = d_nh_domain * d_comm->getSize();

    // Set counters.
    d_nh_left = d_nh_domain;
    d_nh_emitted = 0;

    // Barrier before continuing.
    d_comm->barrier();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get a history from the source.
 */
template<class Domain>
Teuchos::RCP<typename UniformSource<Domain>::HistoryType> 
UniformSource<Domain>::getHistory()
{
    Require( d_weight > 0.0 );
    Require( GlobalRNG::d_rng.assigned() );
    Require( d_nh_left >= 0 );

    // Return null if empty.
    if ( !d_nh_left )
    {
	return Teuchos::null;
    }

    // Generate the history.
    Teuchos::RCP<HistoryType> history = Teuchos::rcp( new HistoryType() );
    history.setRNG( GlobalRNG::d_rng );
    RNG rng = history->rng();

    // Sample the local source vector to get a starting state.
    Teuchos::ArrayRCP<double> local_source = VT::viewNonConst( *Base::b_b );
    Teuchos::ArrayView<double>::size_type local_state =
	SamplingTools::sampleDiscreteCDF( local_source(), rng.random() );
    Ordinal starting_state = VT::globalRow( local_state );

    // Set the history state.
    history->setState( starting_state );
    history->setWeight( d_weight );
    history->live();

    // Update count.
    --d_nh_left;
    ++d_nh_emitted;

    Ensure( !history.is_null() );
    Ensure( history->alive() );
    return history;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Make a globally unique random number generator for this proc.
 *
 * This function creates unique RNGs for each proc so that each history in the
 * parallel domain will sample from a globally unique stream.
 */
template<class Domain>
void UniformSource<Domain>::makeRNG()
{
    GlobalRNG::d_rng = b_rng_control->rng( d_rng_stream + d_comm->getRank() );
    d_rng_stream += d_comm->getSize();

    Ensure( GlobalRNG::d_rng.assigned() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMSOURCE_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformSource_impl.hpp
//---------------------------------------------------------------------------//

