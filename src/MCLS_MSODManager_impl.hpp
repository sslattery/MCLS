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

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
MSODManager<Domain>::MSODManager( const Teuchos::RCP<Domain>& primary_domain,
				  const Teuchos::RCP<const Comm>& global_comm,
				  Teuchos::ParameterList& plist )
    : d_global_comm( global_comm )
    , d_num_sets( plist.get<int>("Number of Sets") )
    , d_set_size( 0 )
    , d_block_size( d_num_sets )
    , d_set_id( -1 )
    , d_block_id( -1 )
{
    Require( !global_comm.is_null() );
    Require( d_num_sets > 0 );

    // Get the set size. We could compute this value from user input, but we
    // must Insist that this is true every time and therefore we do this
    // reduction to verify. We require the primary domain to not exist on
    // procs not owned by the primary domain.
    int local_size = Teuchos::as<int>( !primary_domain.is_null() );
    Teuchos::reduceAll<int,int>( *d_global_comm, Teuchos::REDUCE_SUM,
				 local_size, Teuchos::Ptr<int>(&d_set_size) );
    Insist( d_num_sets * d_set_size == d_global_comm->getSize(),
	    "Size of set * Number of sets != Global communicator size!" );
    Check( d_set_size > 0 );

    // The number of blocks will be equal to the set size.
    d_num_blocks = d_set_size;
    Check( d_num_blocks * d_block_size == d_global_comm->getSize() );

    // We require that the primary domain exist on global procs 0 through
    // (d_set_size-1). If the primary domain exists, it is also the local
    // domain.
    if ( d_global_comm->getRank() < d_set_size )
    {
	Insist( !primary_domain.is_null(),
		"Primary domain must exist on procs [0,(set_size-1)] only!" );

	// The local domain is the primary domain in this case.
	d_local_domain = primary_domain;
    }
    else
    {
	Insist( primary_domain.is_null(),
		"Primary domain must exist on procs [0,(set_size-1)] only!" );
    }

    // Generate the set-constant communicators and the set ids.
    buildSetComms();

    // Generate the block-constant communicators and the block ids.
    buildBlockComms();

    // Pack the primary domain and broadcast across the blocks.
    buildDecomposition();

    // Barrier before proceeding.
    d_global_comm->barrier();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the set-constant commumnicators.
 */
template<class Domain>
void MSODManager<Domain>::buildSetComms()
{
    Require( d_set_size > 0 );
    Require( d_num_sets > 0 );

    Teuchos::Array<int> subcomm_ranks( d_set_size );
    Teuchos::RCP<const Comm> set_comm;
    for ( int i = 0; i < d_num_sets; ++i )
    {
	for ( int n = i*d_set_size; n < (i+1)*d_set_size; ++n )
	{
	    subcomm_ranks[n - i*d_set_size] = n;
	}

	set_comm = d_global_comm->createSubcommunicator( subcomm_ranks() );

	if ( !set_comms.is_null() )
	{
	    d_set_comm = set_comm;
	    d_set_id = i;
	}
    }

    Ensure( !d_set_comm.is_null() );
    Ensure( d_set_id >= 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the block-constant commumnicators.
 */
template<class Domain>
void BlockManager<Domain>::buildBlockComms()
{
    Require( d_block_size > 0 );
    Require( d_num_blocks > 0 );

    Teuchos::Array<int> subcomm_ranks( d_block_size );
    Teuchos::RCP<const Comm> block_comm;
    for ( int i = 0; i < d_num_blocks; ++i )
    {
	for ( int n = i*d_block_size; n < (i+1)*d_block_size; ++n )
	{
	    subcomm_ranks[n - i*d_block_size] = n;
	}

	block_comm = d_global_comm->createSubcommunicator( subcomm_ranks() );

	if ( !block_comms.is_null() )
	{
	    d_block_comm = block_comm;
	    d_block_id = i;
	}
    }

    Ensure( !d_block_comm.is_null() );
    Ensure( d_block_id >= 0 );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the global decomposition by broadcasting the primary domain.
 */
template<class Domain>
void MSODManager<Domain>::buildDecomposition()
{
    Require( !d_block_comm.is_null() );

    Domain::setByteSize();
    Teuchos::Array<char> domain_buffer( Domain::getPackedBytes() );
    if ( !d_local_domain.is_null() )
    {
	domain_buffer = d_local_domain->pack();
	Check( domain_buffer.size() == Domain::getPackedBytes() );
    }

    Teuchos::broadcast<int,char>( *d_block_comm, 0, domain_buffer() );

    if ( d_local_domain.is_null() )
    {
	d_local_domain = Teuchos::rcp( new Domain(domain_buffer()) );
    }

    Ensure( !d_local_domain.is_null() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_MSODMANAGER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MSODManager_impl.hpp
// ---------------------------------------------------------------------------//

