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
#include "MCLS_SamplingTools.hpp"
#include "MCLS_Serializer.hpp"
#include "MCLS_SamplingTools.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>

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
    const Teuchos::ParameterList& plist )
    : d_b( b )
    , d_domain( domain )
    , d_rng_dist( RDT::create(0.0, 1.0) )
    , d_nh_requested( VT::getGlobalLength(*d_b) )
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight( VT::norm1(*d_b) )
    , d_nh_left(0)
    , d_nh_emitted(0)
    , d_random_sampling(1)
    , d_local_length( VT::getLocalLength(*d_b) )
    , d_cdf( d_local_length )
    , d_samples_per_state( d_local_length )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_b) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );

    // Get the requested number of histories. The default value is a sample
    // ratio of 1.
    if ( plist.isParameter("Sample Ratio") )
    {
	d_nh_requested = 
	    VT::getGlobalLength(*d_b) * plist.get<double>("Sample Ratio");
    }
    
    // Determine whether to use random or stratified source sampling. Default
    // to use random sampling.
    if ( plist.isParameter("Source Sampling Type") )
    {
        if ( plist.get<std::string>("Source Sampling Type") == "Random" )
        {
            d_random_sampling = 1;
        }
        else if ( plist.get<std::string>("Source Sampling Type") == 
                  "Stratified" )
        {
            d_random_sampling = 0;
        }
    }

    // Set the total to the requested amount. This may change based on the
    // global stratified sampling.
    d_nh_total = d_nh_requested;

    // Get the global states in the source.
    d_global_states = d_b->getMap()->getNodeElementList();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the source.
 */
template<class Domain>
void UniformAdjointSource<Domain>::buildSource()
{
    // Get the local source components.
    d_local_source = VT::view( *d_b );
    d_local_length = VT::getLocalLength(*d_b);
    MCLS_CHECK( d_local_source.size() > 0 );

    // Build the source.
    if ( d_random_sampling )
    {
        buildRandomSource();
    }
    else
    {
        buildStratifiedSource();
    }

    // The total size may have changed due to integer rounding.
    Teuchos::reduceAll( *VT::getComm(*d_b), Teuchos::REDUCE_SUM, 
			d_nh_domain, Teuchos::Ptr<int>(&d_nh_total) );
    MCLS_CHECK( d_nh_total > 0 );

    // Set counters.
    d_nh_left = d_nh_domain;
    d_nh_emitted = 0;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get a history from the source.
 */
template<class Domain>
typename UniformAdjointSource<Domain>::HistoryType
UniformAdjointSource<Domain>::getHistory()
{
    MCLS_REQUIRE( d_weight > 0.0 );
    MCLS_REQUIRE( d_nh_left > 0 );
    MCLS_REQUIRE( Teuchos::nonnull(d_rng) );

    // Get the next state.
    MCLS_REQUIRE( d_history_stack.top().second > 0 );
    int local_state = d_history_stack.top().first;
    MCLS_CHECK( VT::isLocalRow(*d_b,local_state) );
    
    // Update the state count.
    --d_history_stack.top().second;
    if ( 0 == d_history_stack.top().second )
    {
        d_history_stack.pop();
    }

    // Update count.
    --d_nh_left;
    ++d_nh_emitted;

    // Generate the history.
    Ordinal weight_sign = (d_local_source[local_state] > 0.0) ? 1 : -1;
    return HistoryType(
	d_global_states[local_state], local_state, d_weight * weight_sign );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build a random source.
 */
template<class Domain>
void UniformAdjointSource<Domain>::buildRandomSource()
{
    // Build a non-normalized CDF from the local source data.
    d_cdf[0] = d_local_source[0];
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator src_it;
    Teuchos::ArrayRCP<double>::iterator cdf_it;
    for ( src_it = d_local_source.begin()+1, cdf_it = d_cdf.begin()+1;
	  src_it != d_local_source.end();
	  ++src_it, ++cdf_it )
    {
	*cdf_it = *(cdf_it-1) + std::abs(*src_it);
	MCLS_CHECK( *cdf_it >= 0.0 );
    }

    // Stratify sample the global domain to get the number of histories that
    // will be generated by sampling the local cdf.
    d_nh_domain = d_nh_total * d_cdf().back() / VT::norm1(*d_b);

    // Normalize the CDF.
    for ( cdf_it = d_cdf.begin(); cdf_it != d_cdf.end(); ++cdf_it )
    {
	*cdf_it /= d_cdf().back();
	MCLS_CHECK( *cdf_it >= 0 );
    }
    MCLS_CHECK( std::abs(d_cdf().back()-1) < 1.0e-6 );

    // Randomly sample the source to build the history stack.
    for ( auto& i : d_samples_per_state ) i = 0;
    for ( int i = 0; i < d_nh_domain; ++i )
    {
	++d_samples_per_state[
	    SamplingTools::sampleDiscreteCDF( 
		d_cdf.getRawPtr(), d_cdf.size(), d_rng->random(*d_rng_dist) )
	    ];
    }
    for ( int i = 0; i < d_local_source.size(); ++i )
    {
        if ( d_samples_per_state[i] > 0 )
        {
            d_history_stack.emplace( std::pair<int,int>(i,d_samples_per_state[i]) );
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build a stratified source.
 */
template<class Domain>
void UniformAdjointSource<Domain>::buildStratifiedSource()
{
    // Get the 1-norm of the local source.
    double local_sum = 0.0;
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator src_it;
    for ( src_it = d_local_source.begin(); 
          src_it != d_local_source.end();
	  ++src_it )
    {
	local_sum += std::abs(*src_it);
	MCLS_CHECK( local_sum >= 0 );
    }

    // Stratify sample the global domain to get the number of histories that
    // will be generated locally.
    int nh_local = std::ceil( d_nh_total * local_sum / d_weight );

    // Stratify sample the local domain to get a delayed stack of the number
    // of histories to be generated in each state.
    d_nh_domain = 0;
    int nh_state = 0;
    double num_over_sum = nh_local / local_sum;
    for ( int i = 0; i < d_local_source.size(); ++i )
    {
        nh_state = std::ceil( std::abs(d_local_source[i]) * num_over_sum );

        if ( nh_state > 0 )
        {
            d_history_stack.emplace( std::pair<int,int>(i,nh_state) );
            d_nh_domain += nh_state;
        }
    }
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMADJOINTSOURCE_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformAdjointSource_impl.hpp
//---------------------------------------------------------------------------//

