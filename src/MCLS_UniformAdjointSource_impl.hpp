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
#include <Teuchos_ArrayRCP.hpp>

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
    const Teuchos::RCP<const Comm>& set_comm,
    const int global_comm_size,
    const int global_comm_rank,
    const Teuchos::ParameterList& plist )
    : d_b( b )
    , d_domain( domain )
    , d_rng_dist( RDT::create(0.0, 1.0) )
    , d_set_comm( set_comm )
    , d_global_size( global_comm_size )
    , d_global_rank( global_comm_rank )
    , d_nh_requested( VT::getGlobalLength(*d_b) )
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight( VT::norm1(*d_b) )
    , d_nh_left(0)
    , d_nh_emitted(0)
    , d_random_sampling(1)
{
    MCLS_REQUIRE( Teuchos::nonnull(d_b) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );
    MCLS_REQUIRE( Teuchos::nonnull(d_set_comm) );

    // Get the requested number of histories. The default value is the
    // length of the source vector in this set.
    if ( plist.isParameter("Set Number of Histories") )
    {
	d_nh_requested = plist.get<int>("Set Number of Histories");
    }
    
    // Determine whether to use random or stratified source sampling.
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
}

//---------------------------------------------------------------------------//
/*!
 * \brief Deserializer constructor.
 *
 * Note that the relative weight cutoff will be set in the global parameter
 * list by the first call to the vector constructor above.
 */
template<class Domain>
UniformAdjointSource<Domain>::UniformAdjointSource( 
    const Teuchos::ArrayView<char>& buffer,
    const Teuchos::RCP<Domain>& domain,
    const Teuchos::RCP<const Comm>& set_comm,
    const int global_comm_size,
    const int global_comm_rank )
    : d_domain( domain )
    , d_rng_dist( RDT::create(0.0, 1.0) )
    , d_set_comm( set_comm )
    , d_global_size( global_comm_size )
    , d_global_rank( global_comm_rank )
    , d_nh_requested(0)
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight(0)
    , d_nh_left(0)
    , d_nh_emitted(0)
{
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );
    MCLS_REQUIRE( Teuchos::nonnull(d_set_comm) );

    Deserializer ds;
    ds.setBuffer( buffer() );

    // Unpack the sampling type.
    ds >> d_random_sampling;
    MCLS_CHECK( 0 == d_random_sampling || 1 == d_random_sampling );

    // Unpack the requested number of histories.
    ds >> d_nh_requested;
    MCLS_CHECK( d_nh_requested > 0 );

    // Unpack the size of the local source data.
    std::size_t local_size = 0;
    ds >> local_size;
    MCLS_CHECK( local_size > 0 );

    // Unpack the source global rows.
    Teuchos::ArrayRCP<Ordinal> global_rows( local_size );
    typename Teuchos::ArrayRCP<Ordinal>::iterator row_it;
    for ( row_it = global_rows.begin(); 
	  row_it != global_rows.end();
	  ++row_it )
    {
	ds >> *row_it;
    }

    // Build the source vector.
    d_b = VT::createFromRows( d_set_comm, global_rows() );

    // Unpack the local source data.
    Teuchos::ArrayRCP<Scalar> b_view = VT::viewNonConst( *d_b );
    typename Teuchos::ArrayRCP<Scalar>::iterator data_it;
    for ( data_it = b_view.begin(); data_it != b_view.end(); ++data_it )
    {
	ds >> *data_it;
    }
    MCLS_CHECK( ds.getPtr() == ds.end() );

    // Set the weight.
    d_weight = VT::norm1( *d_b );

    // Set the total to the requested amount. This may change based on the
    // global stratified sampling.
    d_nh_total = d_nh_requested;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the source into a buffer.
 */
template<class Domain>
Teuchos::Array<char> UniformAdjointSource<Domain>::pack() const
{
    // Get the byte size of the buffer.
    std::size_t packed_bytes = getPackedBytes();
    MCLS_CHECK( packed_bytes );

    // Build the buffer and set it with the serializer.
    Teuchos::Array<char> buffer( packed_bytes );
    Serializer s;
    s.setBuffer( buffer() );

    // Pack the sampling type.
    s << d_random_sampling;

    // Pack the requested number of histories.
    s << d_nh_requested;

    // Pack the size of the local source data.
    s << Teuchos::as<std::size_t>( VT::getLocalLength(*d_b) );

    // Pack the source global rows.
    for ( Ordinal i = 0; i < VT::getLocalLength(*d_b); ++i )
    {
	s << Teuchos::as<Ordinal>( VT::getGlobalRow(*d_b,i) );
    }

    // Pack the local source data.
    Teuchos::ArrayRCP<const Scalar> b_view = VT::view(*d_b );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator b_view_it;
    for ( b_view_it = b_view.begin(); b_view_it != b_view.end(); ++b_view_it )
    {
	s << *b_view_it;
    }

    MCLS_ENSURE( s.getPtr() == s.end() );

    return buffer;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Compute the size of this object in bytes.
 */
template<class Domain>
std::size_t UniformAdjointSource<Domain>::getPackedBytes() const
{
    Serializer s;
    s.computeBufferSizeMode();

    // Pack the sampling type.
    s << d_random_sampling;

    // Pack the requested number of histories.
    s << d_nh_requested;

    // Pack the size of the local source data.
    s << Teuchos::as<std::size_t>( VT::getLocalLength(*d_b) );

    // Pack the source global rows.
    for ( Ordinal i = 0; i < VT::getLocalLength(*d_b); ++i )
    {
	s << Teuchos::as<Ordinal>( VT::getGlobalRow(*d_b,i) );
    }

    // Pack the local source data.
    Teuchos::ArrayRCP<const Scalar> b_view = VT::view(*d_b );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator b_view_it;
    for ( b_view_it = b_view.begin(); b_view_it != b_view.end(); ++b_view_it )
    {
	s << *b_view_it;
    }

    return s.size();
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
    Teuchos::reduceAll( *d_set_comm, Teuchos::REDUCE_SUM, 
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
Teuchos::RCP<typename UniformAdjointSource<Domain>::HistoryType> 
UniformAdjointSource<Domain>::getHistory()
{
    MCLS_REQUIRE( d_weight > 0.0 );
    MCLS_REQUIRE( d_nh_left >= 0 );
    MCLS_REQUIRE( Teuchos::nonnull(d_rng) );

    // Return null if empty.
    if ( !d_nh_left )
    {
	return Teuchos::null;
    }

    // Get the local source components.
    Teuchos::ArrayRCP<const Scalar> local_source = VT::view( *d_b );
    MCLS_CHECK( local_source.size() > 0 );

    // Generate the history.
    Teuchos::RCP<HistoryType> history = Teuchos::rcp( new HistoryType() );

    // Sample the local source cdf to get a starting state.
    int local_state = d_random_sampling ? 
                      sampleRandomSource( d_rng->random(*d_rng_dist) ) : 
                      sampleStratifiedSource();
    MCLS_CHECK( VT::isLocalRow(*d_b,local_state) );
    Ordinal starting_state = VT::getGlobalRow( *d_b, local_state );
    MCLS_CHECK( DT::isGlobalState(*d_domain,starting_state) );

    // Set the history state.
    Ordinal weight_sign = (local_source[local_state] > 0.0) -
			  (local_source[local_state] < 0.0);
    history->setWeight( d_weight * weight_sign );
    history->setGlobalState( starting_state );
    history->live();

    // Update count.
    --d_nh_left;
    ++d_nh_emitted;

    MCLS_ENSURE( Teuchos::nonnull(history) );
    MCLS_ENSURE( history->alive() );
    MCLS_ENSURE( history->weightAbs() == d_weight );

    return history;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build a random source.
 */
template<class Domain>
void UniformAdjointSource<Domain>::buildRandomSource()
{
    // Build a non-normalized CDF from the local source data.
    d_cdf = Teuchos::ArrayRCP<double>( d_local_source.size(), 
				       std::abs(d_local_source[0]) );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator src_it;
    Teuchos::ArrayRCP<double>::iterator cdf_it;
    for ( src_it = d_local_source.begin()+1, cdf_it = d_cdf.begin()+1;
	  src_it != d_local_source.end();
	  ++src_it, ++cdf_it )
    {
	*cdf_it = *(cdf_it-1) + std::abs(*src_it);
	MCLS_CHECK( *cdf_it >= 0 );
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
            d_history_stack.push( std::pair<int,int>(i,nh_state) );
            d_nh_domain += nh_state;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Sample the random source.
 */
template<class Domain>
int UniformAdjointSource<Domain>::sampleRandomSource( const double random )
{
    // Sample the local source cdf to get a starting state.
    return SamplingTools::sampleDiscreteCDF( 
	d_cdf.getRawPtr(), d_cdf.size(), random );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Sample the stratified source.
 */
template<class Domain>
int UniformAdjointSource<Domain>::sampleStratifiedSource()
{
    // Get the next state.
    MCLS_REQUIRE( d_history_stack.top().second > 0 );
    int local_state = d_history_stack.top().first;

    // Update the state count.
    --d_history_stack.top().second;
    if ( 0 == d_history_stack.top().second )
    {
        d_history_stack.pop();
    }

    return local_state;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMADJOINTSOURCE_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformAdjointSource_impl.hpp
//---------------------------------------------------------------------------//

