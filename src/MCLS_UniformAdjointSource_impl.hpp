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
UniformAdjointSource<Domain>::UniformAdjointSource( 
    const Teuchos::RCP<VectorType>& b,
    const Teuchos::RCP<Domain>& domain,
    const Teuchos::RCP<RNGControl>& rng_control,
    const Teuchos::RCP<const Comm>& set_comm,
    const int global_comm_size,
    const int global_comm_rank,
    const Teuchos::ParameterList& plist )
    : d_b( b )
    , d_domain( domain )
    , d_rng_control( rng_control )
    , d_set_comm( set_comm )
    , d_global_size( global_comm_size )
    , d_global_rank( global_comm_rank )
    , d_rng_stream(0)
    , d_nh_requested( VT::getGlobalLength(*d_b) )
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight( VT::norm1(*d_b) )
    , d_nh_left(0)
    , d_nh_emitted(0)
{
    Require( !d_b.is_null() );
    Require( !d_domain.is_null() );
    Require( !d_rng_control.is_null() );
    Require( !d_set_comm.is_null() );

    // Get the requested number of histories. The default value is the
    // length of the source vector in this set.
    if ( plist.isParameter("Set Number of Histories") )
    {
	d_nh_requested = plist.get<int>("Set Number of Histories");
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
    const Teuchos::RCP<RNGControl>& rng_control,
    const Teuchos::RCP<const Comm>& set_comm,
    const int global_comm_size,
    const int global_comm_rank )
    : d_domain( domain )
    , d_rng_control( rng_control )
    , d_set_comm( set_comm )
    , d_global_size( global_comm_size )
    , d_global_rank( global_comm_rank )
    , d_rng_stream(0)
    , d_nh_requested(0)
    , d_nh_total(0)
    , d_nh_domain(0)
    , d_weight(0)
    , d_nh_left(0)
    , d_nh_emitted(0)
{
    Require( !d_domain.is_null() );
    Require( !d_rng_control.is_null() );
    Require( !d_set_comm.is_null() );

    Deserializer ds;
    ds.setBuffer( buffer() );

    // Unpack the requested number of histories.
    ds >> d_nh_requested;
    Check( d_nh_requested > 0 );

    // Unpack the size of the local source data.
    std::size_t local_size = 0;
    ds >> local_size;
    Check( local_size > 0 );

    // Unpack the source global rows.
    Teuchos::ArrayRCP<Ordinal> global_rows( local_size );
    typename Teuchos::ArrayRCP<Ordinal>::iterator row_it;
    for ( row_it = global_rows.begin(); 
	  row_it != global_rows.end();
	  ++row_it )
    {
	ds >> *row_it;
    }

    // Unpack the local source data.
    Teuchos::ArrayRCP<Scalar> source_data( local_size );
    typename Teuchos::ArrayRCP<Scalar>::iterator data_it;
    for ( data_it = source_data.begin(); 
	  data_it != source_data.end(); 
	  ++data_it )
    {
	ds >> *data_it;
    }

    Check( ds.getPtr() == ds.end() );
   
    // Build the source vector.
    d_b = VT::createFromRows( d_set_comm, global_rows() );

    // Set the data in the source vector.
    typename Teuchos::ArrayRCP<Scalar>::const_iterator const_data_it;
    for ( const_data_it = source_data.begin(), row_it = global_rows.begin(); 
	  const_data_it != source_data.end(); 
	  ++const_data_it, ++row_it )
    {
	VT::replaceGlobalValue( *d_b, *row_it, *const_data_it );
    }
    source_data.clear();

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
    Check( packed_bytes );

    // Build the buffer and set it with the serializer.
    Teuchos::Array<char> buffer( packed_bytes );
    Serializer s;
    s.setBuffer( buffer() );

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

    Ensure( s.getPtr() == s.end() );

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
    // Set the RNG stream.
    makeRNG();

    // Get the local source components.
    Teuchos::ArrayRCP<const Scalar> local_source = VT::view( *d_b );
    Check( local_source.size() > 0 );

    // Build a non-normalized CDF from the local source data.
    d_cdf = Teuchos::ArrayRCP<double>( local_source.size(), 
				       std::abs(local_source[0]) );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator src_it;
    Teuchos::ArrayRCP<double>::iterator cdf_it;
    for ( src_it = local_source.begin()+1, cdf_it = d_cdf.begin()+1;
	  src_it != local_source.end();
	  ++src_it, ++cdf_it )
    {
	*cdf_it = *(cdf_it-1) + std::abs(*src_it);
	Check( *cdf_it >= 0 );
    }

    // Stratify sample the global domain to get the number of histories that
    // will be generated by sampling the local cdf.
    d_nh_domain = d_nh_total * d_cdf().back() / d_weight;

    // The total size may have changed due to integer rounding.
    Teuchos::reduceAll( *d_set_comm, Teuchos::REDUCE_SUM, 
			d_nh_domain, Teuchos::Ptr<int>(&d_nh_total) );
    Check( d_nh_total > 0 );

    // Normalize the CDF.
    for ( cdf_it = d_cdf.begin(); cdf_it != d_cdf.end(); ++cdf_it )
    {
	*cdf_it /= d_cdf().back();
	Check( *cdf_it >= 0 );
    }
    Check( std::abs(d_cdf().back()-1) < 1.0e-6 );

    // Set counters.
    d_nh_left = d_nh_domain;
    d_nh_emitted = 0;

    // Barrier before continuing.
    d_set_comm->barrier();
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

    // Get the local source components.
    Teuchos::ArrayRCP<const Scalar> local_source = VT::view( *d_b );
    Check( local_source.size() > 0 );

    // Generate the history.
    Teuchos::RCP<HistoryType> history = Teuchos::rcp( new HistoryType() );
    history->setRNG( GlobalRNG::d_rng );
    RNG rng = history->rng();

    // Sample the local source cdf to get a starting state.
    Teuchos::ArrayView<double>::size_type local_state =
	SamplingTools::sampleDiscreteCDF( d_cdf(), rng.random() );
    Ordinal starting_state = VT::getGlobalRow( *d_b, local_state );
    Check( DT::isLocalState(*d_domain,starting_state) );

    // Set the history state.
    Ordinal weight_sign = 
	local_source[local_state] / std::abs(local_source[local_state]);
    history->setWeight( d_weight * weight_sign );
    history->setState( starting_state );
    history->live();

    // Update count.
    --d_nh_left;
    ++d_nh_emitted;

    Ensure( !history.is_null() );
    Ensure( history->alive() );
    Ensure( history->weightAbs() == d_weight );

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
    GlobalRNG::d_rng = d_rng_control->rng( 
	d_rng_stream + d_global_rank );
    d_rng_stream += d_global_size;

    Ensure( GlobalRNG::d_rng.assigned() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMADJOINTSOURCE_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformAdjointSource_impl.hpp
//---------------------------------------------------------------------------//

