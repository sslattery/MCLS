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
 * \file MCLS_UniformAdjointSource_impl.hpp
 * \author Stuart R. Slattery
 * \brief UniformAdjointSource class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_UNIFORMADJOINTSOURCE_IMPL_HPP
#define MCLS_UNIFORMADJOINTSOURCE_IMPL_HPP

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
UniformAdjointSource<Domain>::UniformAdjointSource( 
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
    
    // Set the total to the requested amount. This may change based on the
    // global stratified sampling.
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
void UniformAdjointSource<Domain>::buildSource()
{
    // Set the RNG stream.
    makeRNG();

    // Get the local source components.
    Teuchos::ArrayRCP<const Scalar> local_source = VT::view( *Base::b_b );
    Check( local_source.size() > 0 );

    // Build a cdf from the local source data.
    Teuchos::ArrayRCP<double> d_cdf( local_source.size(), 
				     std::abs(local_source[0]) );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator src_it;
    Teuchos::ArrayRCP<double>::iterator cdf_it;
    for ( src_it = local_source.begin()+1, cdf_it = d_cdf.begin()+1;
	  src_it != local_source.end();
	  ++src_it, ++cdf_it )
    {
	*cdf_it = *(cdf_it-1) + std::abs(*src_it);
    }

    // Stratify sample the global domain to get the number of histories that
    // will be generated by sampling to local cdf.
    d_nh_domain = d_nh_total * d_cdf().back() / d_weight;

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
Teuchos::RCP<typename UniformAdjointSource<Domain>::HistoryType> 
UniformAdjointSource<Domain>::getHistory()
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

    // Sample the local source cdf to get a starting state.
    Teuchos::ArrayView<double>::size_type local_state =
	SamplingTools::sampleDiscreteCDF( d_cdf(), rng.random() );
    Ordinal starting_state = VT::getGlobalRow( *Base::b_b, local_state );

    Check( d_domain->isLocalState(starting_state) );

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
void UniformAdjointSource<Domain>::makeRNG()
{
    GlobalRNG::d_rng = b_rng_control->rng( d_rng_stream + d_comm->getRank() );
    d_rng_stream += d_comm->getSize();

    Ensure( GlobalRNG::d_rng.assigned() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMADJOINTSOURCE_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformAdjointSource_impl.hpp
//---------------------------------------------------------------------------//

