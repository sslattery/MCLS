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
 * \file MCLS_MSODManager_impl.hpp
 * \author Stuart R. Slattery
 * \brief Multiple-set overlapping-domain decomposition manager implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MSODMANAGER_IMPL_HPP
#define MCLS_MSODMANAGER_IMPL_HPP

#include "MCLS_DBC.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_as.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Source>
MSODManager<Source>::MSODManager( const bool primary_set,
				  const Teuchos::RCP<const Comm>& global_comm,
				  const Teuchos::ParameterList& plist )
    : d_global_comm( global_comm )
    , d_num_sets( plist.get<int>("Number of Sets") )
    , d_num_blocks( 0 )
    , d_set_size( 0 )
    , d_block_size( d_num_sets )
    , d_set_id( -1 )
    , d_block_id( -1 )
{
    MCLS_REQUIRE( !d_global_comm.is_null() );
    MCLS_REQUIRE( d_num_sets > 0 );

    // Get the set size. We could compute this value from user input, but we
    // must Insist that this is true every time and therefore we do this
    // reduction to verify. We require the primary domain to not exist on
    // procs not owned by the primary domain.
    int local_size = Teuchos::as<int>( primary_set );
    Teuchos::reduceAll<int,int>( *d_global_comm, Teuchos::REDUCE_SUM,
				 local_size, Teuchos::Ptr<int>(&d_set_size) );
    MCLS_INSIST( d_num_sets * d_set_size == d_global_comm->getSize(),
                 "Size of set * Number of sets != Global communicator size!" );
    MCLS_CHECK( d_set_size > 0 );

    // The number of blocks will be equal to the set size.
    d_num_blocks = d_set_size;
    MCLS_CHECK( d_num_blocks * d_block_size == d_global_comm->getSize() );

    // We require that the primary domain exist on global procs 0 through
    // (d_set_size-1). If the primary domain exists, it is also the local
    // domain. The same requirements are also applied to the primary source.
    if ( d_global_comm->getRank() < d_set_size )
    {
	MCLS_INSIST( primary_set,
                     "Primary set must exist on procs [0,(set_size-1)] only!" );
    }
    else
    {
	MCLS_INSIST( !primary_set,
                     "Primary set must exist on procs [0,(set_size-1)] only!" );
    }

    // Compute the set id.
    d_set_id = std::floor( Teuchos::as<double>(d_global_comm->getRank()) /
                           Teuchos::as<double>(d_set_size) );
    MCLS_CHECK( d_set_id >=0 && d_set_id < d_num_sets );

    // Compute the block id.
    d_block_id = d_global_comm->getRank() - d_num_blocks*d_set_id;
    MCLS_CHECK( d_block_id >=0 && d_block_id < d_num_blocks );

    // Barrier before proceeding.
    d_global_comm->barrier();

    // Generate the set-constant communicators.
    d_set_comm = d_global_comm->split( d_set_id, d_block_id );

    // Generate the block-constant communicators.
    d_block_comm = d_global_comm->split( d_block_id, d_set_id );

    // Barrier before proceeding.
    d_global_comm->barrier();

    MCLS_ENSURE( Teuchos::nonnull(d_set_comm) );
    MCLS_ENSURE( Teuchos::nonnull(d_block_comm) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the local domain.
 */
template<class Source>
void MSODManager<Source>::setDomain(
    const Teuchos::RCP<Domain>& primary_domain )
{
    if ( d_set_id == 0 )
    {
	MCLS_INSIST( !primary_domain.is_null(),
                     "Primary domain must exist on set 0!" );

	d_local_domain = primary_domain;
    }
    else
    {
	MCLS_INSIST( primary_domain.is_null(),
                     "Primary domain must exist on set 0 only!" );
    }

    broadcastDomain();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Set the local source.
 */
template<class Source>
void MSODManager<Source>::setSource( 
    const Teuchos::RCP<Source>& primary_source,
    const Teuchos::RCP<RNGControl>& rng_control )
{
    MCLS_REQUIRE( !rng_control.is_null() );

    if ( d_set_id == 0 )
    {
	MCLS_INSIST( !primary_source.is_null(),
                     "Primary source must exist on set 0!" );

	d_local_source = primary_source;
    }
    else
    {
	MCLS_INSIST( primary_source.is_null(),
                     "Primary source must exist on set 0 only!" );
    }

    broadcastSource( rng_control );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the global decomposition by broadcasting the primary domain.
 */
template<class Source>
void MSODManager<Source>::broadcastDomain()
{
    MCLS_REQUIRE( !d_set_comm.is_null() );
    MCLS_REQUIRE( !d_block_comm.is_null() );
    MCLS_REQUIRE( d_set_id >= 0 );

    // Get the byte size of the domain from the primary set.
    std::size_t buffer_size = 0;
    if ( d_set_id == 0 )
    {
	MCLS_CHECK( !d_local_domain.is_null() );
	buffer_size = DT::getPackedBytes( *d_local_domain );
    }
    d_block_comm->barrier();

    // Broadcast the buffer size across the blocks.
    Teuchos::broadcast<int,std::size_t>( 
	*d_block_comm, 0, Teuchos::Ptr<std::size_t>(&buffer_size) );
    MCLS_CHECK( buffer_size > 0 );

    // Pack the primary domain.
    Teuchos::Array<char> domain_buffer( buffer_size );
    if ( d_set_id == 0 )
    {
	MCLS_CHECK( !d_local_domain.is_null() );
	domain_buffer = DT::pack( *d_local_domain );
	MCLS_CHECK( Teuchos::as<std::size_t>(domain_buffer.size()) == buffer_size );
    }
    d_block_comm->barrier();

    // Broadcast the domain across the blocks.
    Teuchos::broadcast<int,char>( *d_block_comm, 0, domain_buffer() );

    // Assign the domain.
    d_local_domain = DT::createFromBuffer( d_set_comm, domain_buffer() );

    // Barrier before continuing.
    d_block_comm->barrier();

    MCLS_ENSURE( !d_local_domain.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the global decomposition by broadcasting the primary source.
 */
template<class Source>
void MSODManager<Source>::broadcastSource( 
    const Teuchos::RCP<RNGControl>& rng_control )
{
    MCLS_REQUIRE( !rng_control.is_null() );
    MCLS_REQUIRE( !d_set_comm.is_null() );
    MCLS_REQUIRE( !d_block_comm.is_null() );
    MCLS_REQUIRE( !d_local_domain.is_null() );
    MCLS_REQUIRE( d_set_id >= 0 ); 

    // Get the byte size of the source from the primary set.
    std::size_t buffer_size = 0;
    if ( d_set_id == 0 )
    {
	MCLS_CHECK( !d_local_source.is_null() );
	buffer_size = ST::getPackedBytes( *d_local_source );
    }
    d_block_comm->barrier();

    // Broadcast the buffer size across the blocks.
    Teuchos::broadcast<int,std::size_t>( 
	*d_block_comm, 0, Teuchos::Ptr<std::size_t>(&buffer_size) );
    MCLS_CHECK( buffer_size > 0 );

    // Pack the primary source.
    Teuchos::Array<char> source_buffer( buffer_size );
    if ( d_set_id == 0 )
    {
	MCLS_CHECK( !d_local_source.is_null() );
	source_buffer = ST::pack( *d_local_source );
	MCLS_CHECK( Teuchos::as<std::size_t>(source_buffer.size()) == buffer_size );
    }
    d_block_comm->barrier();

    // Broadcast the source across the blocks.
    Teuchos::broadcast<int,char>( *d_block_comm, 0, source_buffer() );

    // Assign the source.
    d_local_source = ST::createFromBuffer( source_buffer(),
					   d_set_comm,
					   d_local_domain,
					   rng_control,
					   d_global_comm->getSize(),
					   d_global_comm->getRank() );

    // Barrier before continuing.
    d_block_comm->barrier();

    MCLS_ENSURE( !d_local_source.is_null() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_MSODMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MSODManager_impl.hpp
//---------------------------------------------------------------------------//

