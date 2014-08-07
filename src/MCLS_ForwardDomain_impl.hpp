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
 * \file MCLS_ForwardDomain_impl.hpp
 * \author Stuart R. Slattery
 * \brief ForwardDomain implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FORWARDDOMAIN_IMPL_HPP
#define MCLS_FORWARDDOMAIN_IMPL_HPP

#include <algorithm>
#include <limits>
#include <string>

#include "MCLS_Serializer.hpp"
#include "MCLS_Estimators.hpp"
#include "MCLS_VectorExport.hpp"

#include <Teuchos_as.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Distributor.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Matrix constructor.
 */
template<class Vector, class Matrix, class RNG>
ForwardDomain<Vector,Matrix,RNG>::ForwardDomain( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::ParameterList& plist )
    : d_rng_dist( RDT::create(0.0, 1.0) )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    MCLS_REQUIRE( Teuchos::nonnull(x) );

    // Get the estimator type. User the collision estimator as the default.
    d_estimator = Estimator::COLLISION;
    if ( plist.isParameter("Estimator Type") )
    {
	if ( "Collision" == plist.get<std::string>("Estimator Type") )
	{
	    d_estimator = Estimator::COLLISION;
	}
    }

    // Get the amount of overlap.
    int num_overlap = plist.get<int>( "Overlap Size" );
    MCLS_REQUIRE( num_overlap >= 0 );

    // Build the reduced domain.
    {
        // Generate the overlap for the operator.
        Teuchos::RCP<Matrix> A_overlap;
        if ( num_overlap > 0 )
        {
            A_overlap = 
                MT::copyNearestNeighbors( *A, num_overlap );
        }

        // Get the total number of local rows.
        int num_rows = MT::getLocalNumRows( *A );
        if ( num_overlap > 0 )
        { 
            num_rows += MT::getLocalNumRows( *A_overlap );
        }

        // Allocate space in local row data arrays.
        d_global_columns = 
            Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > >( num_rows );
        d_local_columns = 
            Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<int> > >( num_rows );
        d_cdfs = Teuchos::ArrayRCP<Teuchos::Array<double> >( num_rows );
        d_weights = Teuchos::ArrayRCP<double>( num_rows );
        d_h = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );

        // Build the local CDFs and weights.
	double relaxation = 1.0;
	if ( plist.isParameter("Neumann Relaxation") )
	{
	    relaxation = plist.get<double>("Neumann Relaxation");
	}
        addMatrixToDomain( A, relaxation );
        if ( num_overlap > 0 )
        {
            addMatrixToDomain( A_overlap, relaxation );
        }

        // Get the boundary states and their owning process ranks.
        if ( num_overlap == 0 )
        {
            buildBoundary( A, A );
        }
        else
        {
            buildBoundary( A_overlap, A );
        }
    }

    // Make the set of local columns. If the local column is not a global row
    // then make it invalid to indicate that we have left the domain.
    typename Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator global_it;
    typename Teuchos::Array<Ordinal>::const_iterator gcol_it;
    Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<int> > >::iterator local_it;
    Teuchos::Array<int>::iterator lcol_it;
    for ( global_it = d_global_columns.begin(),
	   local_it = d_local_columns.begin();
	  global_it != d_global_columns.end();
	  ++global_it, ++local_it )
    {
	*local_it = Teuchos::rcp(
	    new Teuchos::Array<int>((*global_it)->size()) );
	for ( gcol_it = (*global_it)->begin(),
	      lcol_it = (*local_it)->begin();
	      gcol_it != (*global_it)->end();
	      ++gcol_it, ++lcol_it )
	{
	    if ( d_g2l_row_indexer.count(*gcol_it) )
	    {
		*lcol_it = d_g2l_row_indexer.find( *gcol_it )->second;
	    }
	    else
	    {
		*lcol_it = Teuchos::OrdinalTraits<int>::invalid();
	    }
	}
    }

    // By building the boundary data, now we know where we are sending
    // data. Find out who we are receiving from.
    Tpetra::Distributor distributor( MT::getComm(*A) );
    distributor.createFromSends( d_send_ranks() );
    d_receive_ranks = distributor.getImagesFrom();

    // Create the tally.
    d_tally = Teuchos::rcp( new TallyType(x, d_estimator) );

    MCLS_ENSURE( Teuchos::nonnull(d_tally) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Deserializer constructor.
 * 
 * \param buffer Data buffer to construct the domain from.
 *
 * \param set_comm Set constant communicator for this domain over which to
 * reconstruct the tallies.
 */
template<class Vector, class Matrix, class RNG>
ForwardDomain<Vector,Matrix,RNG>::ForwardDomain( 
    const Teuchos::ArrayView<char>& buffer,
    const Teuchos::RCP<const Comm>& set_comm )
    : d_rng_dist( RDT::create(0.0, 1.0) )
{
    Ordinal num_rows = 0;
    int num_receives = 0;
    int num_sends = 0;
    Ordinal num_bnd = 0;
    Ordinal num_base = 0;
    Ordinal num_tally = 0;

    Deserializer ds;
    ds.setBuffer( buffer() );

    // Unpack the estimator type.
    ds >> d_estimator;
    MCLS_CHECK( d_estimator >= 0 );

    // Unpack the local number of rows.
    ds >> num_rows;
    MCLS_CHECK( num_rows > 0 );

    // Unpack the number of receive neighbors.
    ds >> num_receives;
    MCLS_CHECK( num_receives >= 0 );

    // Unpack the number of send neighbors.
    ds >> num_sends;
    MCLS_CHECK( num_sends >= 0 );

    // Unpack the number of boundary states.
    ds >> num_bnd;
    MCLS_CHECK( num_bnd >= 0 );

    // Unpack the number of base rows in the tally.
    ds >> num_base;
    MCLS_CHECK( num_base > 0 );

    // Unpack the local row indexer by key-value pairs.
    Ordinal global_row = 0;
    int local_row = 0;
    for ( Ordinal n = 0; n < num_rows; ++n )
    {
	ds >> global_row >> local_row;
	d_g2l_row_indexer[global_row] = local_row;
    }

    // Unpack the local columns.
    std::set<Ordinal> im_unique_cols;
    d_global_columns = 
        Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > >( num_rows );
    Ordinal num_cols = 0;
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::iterator column_it;
    typename Teuchos::Array<Ordinal>::iterator index_it;
    for( column_it = d_global_columns.begin(); 
	 column_it != d_global_columns.end(); 
	 ++column_it )
    {
	// Unpack the number of column entries in the row.
	ds >> num_cols;
        *column_it = Teuchos::rcp( new Teuchos::Array<Ordinal>(num_cols) );

	// Unpack the column indices.
	for ( index_it = (*column_it)->begin();
	      index_it != (*column_it)->end();
	      ++index_it )
	{
	    ds >> *index_it;
            im_unique_cols.insert(*index_it);
	}
    }

    // Unpack the local cdfs.
    d_cdfs = Teuchos::ArrayRCP<Teuchos::Array<double> >( num_rows );
    Ordinal num_values = 0;
    Teuchos::ArrayRCP<Teuchos::Array<double> >::iterator cdf_it;
    Teuchos::Array<double>::iterator value_it;
    for( cdf_it = d_cdfs.begin(); cdf_it != d_cdfs.end(); ++cdf_it )
    {
	// Unpack the number of entries in the row cdf.
	ds >> num_values;
	cdf_it->resize( num_values );

	// Unpack the cdf values.
	for ( value_it = cdf_it->begin();
	      value_it != cdf_it->end();
	      ++value_it )
	{
	    ds >> *value_it;
	}
    }

    // Unpack the iteration matrix values.
    d_h = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );
    num_values = 0;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::iterator h_it;
    Teuchos::ArrayRCP<double>::iterator h_val_it;
    for( h_it = d_h.begin(); h_it != d_h.end(); ++h_it )
    {
        // Unpack the number of entries in the row h.
        ds >> num_values;
        *h_it = Teuchos::ArrayRCP<double>( num_values );

        // Unpack the iteration matrix values.
        for ( h_val_it = h_it->begin();
              h_val_it != h_it->end();
              ++h_val_it )
        {
            ds >> *h_val_it;
        }
    }

    // Unpack the local weights.
    d_weights = Teuchos::ArrayRCP<double>( num_rows );
    Teuchos::ArrayRCP<double>::iterator weight_it;
    for ( weight_it = d_weights.begin();
	  weight_it != d_weights.end();
	  ++weight_it )
    {
	ds >> *weight_it;
    }

    // Unpack the receive ranks.
    d_receive_ranks.resize( num_receives );
    Teuchos::Array<int>::iterator receive_it;
    for ( receive_it = d_receive_ranks.begin();
	  receive_it != d_receive_ranks.end();
	  ++receive_it )
    {
	ds >> *receive_it;
    }

    // Unpack the send ranks.
    d_send_ranks.resize( num_sends );
    Teuchos::Array<int>::iterator send_it;
    for ( send_it = d_send_ranks.begin();
	  send_it != d_send_ranks.end();
	  ++send_it )
    {
	ds >> *send_it;
    }

    // Unpack the boundary-to-neighbor id table.
    Ordinal boundary_row = 0;
    int neighbor = 0;
    for ( Ordinal n = 0; n < num_bnd; ++n )
    {
	ds >> boundary_row >> neighbor;
	d_bnd_to_neighbor[boundary_row] = neighbor;
    }

    // Unpack the tally base rows.
    Teuchos::Array<Ordinal> base_rows( num_base );
    typename Teuchos::Array<Ordinal>::iterator base_it;
    for ( base_it = base_rows.begin();
	  base_it != base_rows.end();
	  ++base_it )
    {
	ds >> *base_it;
    }

    // Make the set of local columns. If the local column is not a global row
    // then make it invalid to indicate that we have left the domain.
    d_local_columns = 
        Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<int> > >( num_rows );
    typename Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator global_it;
    typename Teuchos::Array<Ordinal>::const_iterator gcol_it;
    Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<int> > >::iterator local_it;
    Teuchos::Array<int>::iterator lcol_it;
    for ( global_it = d_global_columns.begin(),
	   local_it = d_local_columns.begin();
	  global_it != d_global_columns.end();
	  ++global_it, ++local_it )
    {
	*local_it = Teuchos::rcp(
	    new Teuchos::Array<int>((*global_it)->size()) );
	for ( gcol_it = (*global_it)->begin(),
	      lcol_it = (*local_it)->begin();
	      gcol_it != (*global_it)->end();
	      ++gcol_it, ++lcol_it )
	{
	    if ( d_g2l_row_indexer.count(*gcol_it) )
	    {
		*lcol_it = d_g2l_row_indexer.find( *gcol_it )->second;
	    }
	    else
	    {
		*lcol_it = Teuchos::OrdinalTraits<int>::invalid();
	    }
	}
    }

    // Create the tally.
    Teuchos::RCP<Vector> base_x = 
	VT::createFromRows( set_comm, base_rows() );
    d_tally = Teuchos::rcp( new TallyType(base_x, d_estimator) );

    MCLS_ENSURE( Teuchos::nonnull(d_tally) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the domain into a buffer.
 */
template<class Vector, class Matrix, class RNG>
Teuchos::Array<char> ForwardDomain<Vector,Matrix,RNG>::pack() const
{
    // Get the byte size of the buffer.
    std::size_t packed_bytes = getPackedBytes();
    MCLS_CHECK( packed_bytes );

    // Build the buffer and set it with the serializer.
    Teuchos::Array<char> buffer( packed_bytes );
    Serializer s;
    s.setBuffer( buffer() );

    // Pack the estimator type.
    s << Teuchos::as<int>(d_estimator);

    // Pack the local number of rows.
    s << Teuchos::as<Ordinal>(d_g2l_row_indexer.size());

    // Pack in the number of receive neighbors.
    s << Teuchos::as<int>(d_receive_ranks.size());

    // Pack in the number of send neighbors.
    s << Teuchos::as<int>(d_send_ranks.size());

    // Pack in the number of boundary states.
    s << Teuchos::as<Ordinal>(d_bnd_to_neighbor.size());

    // Pack in the number of base rows in the tally.
    s << Teuchos::as<Ordinal>(d_tally->numBaseRows());

    // Pack up the local row indexer by key-value pairs.
    typename std::unordered_map<Ordinal,int>::const_iterator row_index_it;
    for ( row_index_it = d_g2l_row_indexer.begin();
	  row_index_it != d_g2l_row_indexer.end();
	  ++row_index_it )
    {
	s << row_index_it->first << row_index_it->second;
    }

    // Pack up the local columns.
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator column_it;
    typename Teuchos::Array<Ordinal>::const_iterator index_it;
    for( column_it = d_global_columns.begin(); 
	 column_it != d_global_columns.end(); 
	 ++column_it )
    {
	// Pack the number of column entries in the row.
	s << Teuchos::as<Ordinal>( (*column_it)->size() );

	// Pack in the column indices.
	for ( index_it = (*column_it)->begin();
	      index_it != (*column_it)->end();
	      ++index_it )
	{
	    s << *index_it;
	}
    }

    // Pack up the local cdfs.
    Teuchos::ArrayRCP<Teuchos::Array<double> >::const_iterator cdf_it;
    Teuchos::Array<double>::const_iterator value_it;
    for( cdf_it = d_cdfs.begin(); cdf_it != d_cdfs.end(); ++cdf_it )
    {
	// Pack the number of entries in the row cdf.
	s << Teuchos::as<Ordinal>( cdf_it->size() );

	// Pack in the column indices.
	for ( value_it = cdf_it->begin();
	      value_it != cdf_it->end();
	      ++value_it )
	{
	    s << *value_it;
	}
    }

    // Pack the iteration matrix values.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::const_iterator h_it;
    Teuchos::ArrayRCP<double>::const_iterator h_val_it;
    for( h_it = d_h.begin(); h_it != d_h.end(); ++h_it )
    {
        // Pack the number of entries in the row h.
        s << Teuchos::as<Ordinal>( h_it->size() );

        // Pack the iteration matrix values.
        for ( h_val_it = h_it->begin();
              h_val_it != h_it->end();
              ++h_val_it )
        {
            s << *h_val_it;
        }
    }

    // Pack up the local weights.
    Teuchos::ArrayRCP<double>::const_iterator weight_it;
    for ( weight_it = d_weights.begin();
	  weight_it != d_weights.end();
	  ++weight_it )
    {
	s << *weight_it;
    }

    // Pack up the receive ranks.
    Teuchos::Array<int>::const_iterator receive_it;
    for ( receive_it = d_receive_ranks.begin();
	  receive_it != d_receive_ranks.end();
	  ++receive_it )
    {
	s << *receive_it;
    }

    // Pack up the send ranks.
    Teuchos::Array<int>::const_iterator send_it;
    for ( send_it = d_send_ranks.begin();
	  send_it != d_send_ranks.end();
	  ++send_it )
    {
	s << *send_it;
    }

    // Pack up the boundary-to-neighbor id table.
    typename std::unordered_map<Ordinal,int>::const_iterator bnd_it;
    for ( bnd_it = d_bnd_to_neighbor.begin();
	  bnd_it != d_bnd_to_neighbor.end();
	  ++bnd_it )
    {
	s << bnd_it->first << bnd_it->second;
    }

    // Pack up the tally base rows.
    Teuchos::Array<Ordinal> base_rows = d_tally->baseRows();
    typename Teuchos::Array<Ordinal>::const_iterator base_it;
    for ( base_it = base_rows.begin();
	  base_it != base_rows.end();
	  ++base_it )
    {
	s << *base_it;
    }

    MCLS_ENSURE( s.end() == s.getPtr() );

    return buffer;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size of this object in packed bytes.
 */
template<class Vector, class Matrix, class RNG>
std::size_t ForwardDomain<Vector,Matrix,RNG>::getPackedBytes() const
{
    Serializer s;
    s.computeBufferSizeMode();

    // Pack the estimator type.
    s << Teuchos::as<int>(d_estimator);

    // Pack the local number of rows.
    s << Teuchos::as<Ordinal>(d_g2l_row_indexer.size());

    // Pack in the number of receive neighbors.
    s << Teuchos::as<int>(d_receive_ranks.size());

    // Pack in the number of send neighbors.
    s << Teuchos::as<int>(d_send_ranks.size());

    // Pack in the number of boundary states.
    s << Teuchos::as<Ordinal>(d_bnd_to_neighbor.size());

    // Pack in the number of base rows in the tally.
    s << Teuchos::as<Ordinal>(d_tally->numBaseRows());

    // Pack up the local row indexer by key-value pairs.
    typename std::unordered_map<Ordinal,int>::const_iterator row_index_it;
    for ( row_index_it = d_g2l_row_indexer.begin();
	  row_index_it != d_g2l_row_indexer.end();
	  ++row_index_it )
    {
	s << row_index_it->first << row_index_it->second;
    }

    // Pack up the local columns.
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator column_it;
    typename Teuchos::Array<Ordinal>::const_iterator index_it;
    for( column_it = d_global_columns.begin(); 
	 column_it != d_global_columns.end(); 
	 ++column_it )
    {
	// Pack the number of column entries in the row.
	s << Teuchos::as<Ordinal>( (*column_it)->size() );

	// Pack in the column indices.
	for ( index_it = (*column_it)->begin();
	      index_it != (*column_it)->end();
	      ++index_it )
	{
	    s << *index_it;
	}
    }

    // Pack up the local cdfs.
    Teuchos::ArrayRCP<Teuchos::Array<double> >::const_iterator cdf_it;
    Teuchos::Array<double>::const_iterator value_it;
    for( cdf_it = d_cdfs.begin(); cdf_it != d_cdfs.end(); ++cdf_it )
    {
	// Pack the number of entries in the row cdf.
	s << Teuchos::as<Ordinal>( cdf_it->size() );

	// Pack in the column indices.
	for ( value_it = cdf_it->begin();
	      value_it != cdf_it->end();
	      ++value_it )
	{
	    s << *value_it;
	}
    }

    // Pack the iteration matrix values.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::const_iterator h_it;
    Teuchos::ArrayRCP<double>::const_iterator h_val_it;
    for( h_it = d_h.begin(); h_it != d_h.end(); ++h_it )
    {
        // Pack the number of entries in the row h.
        s << Teuchos::as<Ordinal>( h_it->size() );

        // Pack the iteration matrix values.
        for ( h_val_it = h_it->begin();
              h_val_it != h_it->end();
              ++h_val_it )
        {
            s << *h_val_it;
        }
    }

    // Pack up the local weights.
    Teuchos::ArrayRCP<double>::const_iterator weight_it;
    for ( weight_it = d_weights.begin();
	  weight_it != d_weights.end();
	  ++weight_it )
    {
	s << *weight_it;
    }

    // Pack up the receive ranks.
    Teuchos::Array<int>::const_iterator receive_it;
    for ( receive_it = d_receive_ranks.begin();
	  receive_it != d_receive_ranks.end();
	  ++receive_it )
    {
	s << *receive_it;
    }

    // Pack up the send ranks.
    Teuchos::Array<int>::const_iterator send_it;
    for ( send_it = d_send_ranks.begin();
	  send_it != d_send_ranks.end();
	  ++send_it )
    {
	s << *send_it;
    }

    // Pack up the boundary-to-neighbor id table.
    typename std::unordered_map<Ordinal,int>::const_iterator bnd_it;
    for ( bnd_it = d_bnd_to_neighbor.begin();
	  bnd_it != d_bnd_to_neighbor.end();
	  ++bnd_it )
    {
	s << bnd_it->first << bnd_it->second;
    }

    // Pack up the tally base rows.
    Teuchos::Array<Ordinal> base_rows = d_tally->baseRows();
    typename Teuchos::Array<Ordinal>::const_iterator base_it;
    for ( base_it = base_rows.begin();
	  base_it != base_rows.end();
	  ++base_it )
    {
	s << *base_it;
    }

    return s.size();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the local states owned by this domain.
 */
template<class Vector, class Matrix, class RNG>
Teuchos::Array<typename ForwardDomain<Vector,Matrix,RNG>::Ordinal> 
ForwardDomain<Vector,Matrix,RNG>::localStates() const
{
    Teuchos::Array<Ordinal> states( d_g2l_row_indexer.size() );
    typename Teuchos::Array<Ordinal>::iterator state_it;
    typename std::unordered_map<Ordinal,int>::const_iterator map_it;
    for ( map_it = d_g2l_row_indexer.begin(), state_it = states.begin();
          map_it != d_g2l_row_indexer.end(); 
          ++map_it, ++state_it )
    {
        *state_it = map_it->first;
    }

    return states;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank from which we will receive.
 */
template<class Vector, class Matrix, class RNG>
int ForwardDomain<Vector,Matrix,RNG>::receiveNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < d_receive_ranks.size() );
    return d_receive_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank to which we will send.
 */
template<class Vector, class Matrix, class RNG>
int ForwardDomain<Vector,Matrix,RNG>::sendNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < d_send_ranks.size() );
    return d_send_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain that owns a boundary state (local neighbor
 * id).
 */
template<class Vector, class Matrix, class RNG>
int ForwardDomain<Vector,Matrix,RNG>::owningNeighbor( const Ordinal& state ) const
{
    typename std::unordered_map<Ordinal,int>::const_iterator neighbor = 
	d_bnd_to_neighbor.find( state );
    MCLS_REQUIRE( neighbor != d_bnd_to_neighbor.end() );
    return neighbor->second;
}

//---------------------------------------------------------------------------//
/*
 * \brief Add matrix data to the local domain.
 */
template<class Vector, class Matrix, class RNG>
void ForwardDomain<Vector,Matrix,RNG>::addMatrixToDomain( 
    const Teuchos::RCP<const Matrix>& A,
    const double relaxation )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );

    Ordinal local_num_rows = MT::getLocalNumRows( *A );
    Ordinal global_row = 0;
    int offset = d_g2l_row_indexer.size();
    int ipoffset = 0;
    int max_entries = MT::getGlobalMaxNumRowEntries( *A );
    std::size_t num_entries = 0;
    Teuchos::Array<double>::iterator cdf_iterator;

    for ( Ordinal i = 0; i < local_num_rows; ++i )
    {
	// Get the offset row index.
	ipoffset = i+offset;

	// Add the global row id and local row id to the indexer.
	global_row = MT::getGlobalRow(*A, i);
	d_g2l_row_indexer[global_row] = ipoffset;

	// Allocate column and CDF memory for this row.
        d_global_columns[ipoffset] = 
            Teuchos::rcp( new Teuchos::Array<Ordinal>(max_entries) );
	d_cdfs[ipoffset].resize( max_entries );

	// Add the columns and base PDF values for this row.
	MT::getGlobalRowCopy( *A, 
			      global_row,
			      (*d_global_columns[ipoffset])(), 
			      d_cdfs[ipoffset](),
			      num_entries );

	// Check for degeneracy.
	MCLS_CHECK( num_entries > 0 );

	// Resize local column and CDF arrays for this row.
	d_global_columns[ipoffset]->resize( num_entries );
	d_cdfs[ipoffset].resize( num_entries );

	// Create the iteration matrix.
	for ( std::size_t j = 0; j < num_entries; ++j )
	{
	    // Subtract the operator from the identity matrix.
	    d_cdfs[ipoffset][j] = 
		( (*d_global_columns[ipoffset])[j] == global_row ) ?
		1.0 - relaxation*d_cdfs[ipoffset][j] : 
		-relaxation*d_cdfs[ipoffset][j];

	    // Mark any zero entries.
	    if ( std::abs(d_cdfs[ipoffset][j]) < 
		 std::numeric_limits<double>::epsilon() )
	    {
		d_cdfs[ipoffset][j] = std::numeric_limits<double>::max();
		(*d_global_columns[ipoffset])[j] = 
		    Teuchos::OrdinalTraits<Ordinal>::invalid();
	    }
	}

  	// Extract any zero entries from the iteration matrix.
	Teuchos::Array<double>::iterator cdf_remove_it;
	cdf_remove_it = std::remove( d_cdfs[ipoffset].begin(), 
				     d_cdfs[ipoffset].end(),
				     std::numeric_limits<double>::max() );
	d_cdfs[ipoffset].resize( 
	    std::distance(d_cdfs[ipoffset].begin(), cdf_remove_it) );

	typename Teuchos::Array<Ordinal>::iterator col_remove_it;
	col_remove_it = std::remove( d_global_columns[ipoffset]->begin(), 
				     d_global_columns[ipoffset]->end(),
				     Teuchos::OrdinalTraits<Ordinal>::invalid() );
	d_global_columns[ipoffset]->resize( 
	    std::distance(d_global_columns[ipoffset]->begin(), col_remove_it) );

	// Save the current cdf state as the iteration matrix.
        d_h[ipoffset] = Teuchos::ArrayRCP<double>( d_cdfs[ipoffset].size() );
        std::copy( d_cdfs[ipoffset].begin(), d_cdfs[ipoffset].end(),
                   d_h[ipoffset].begin() );

	// Accumulate the absolute value of the PDF values to get a
	// non-normalized CDF for the row.
	d_cdfs[ipoffset].front() = std::abs( d_cdfs[ipoffset].front() );
	for ( cdf_iterator = d_cdfs[ipoffset].begin()+1;
	      cdf_iterator != d_cdfs[ipoffset].end();
	      ++cdf_iterator )
	{
	    *cdf_iterator = std::abs( *cdf_iterator ) + *(cdf_iterator-1);
	}

	// The final value in the non-normalized CDF is the weight for this
	// row. This is the absolute value row sum of the iteration matrix.
	d_weights[ipoffset] = d_cdfs[ipoffset].back();
	MCLS_CHECK( d_weights[ipoffset] > 0.0 );

	// Normalize the CDF for the row.
	for ( cdf_iterator = d_cdfs[ipoffset].begin();
	      cdf_iterator != d_cdfs[ipoffset].end();
	      ++cdf_iterator )
	{
	    *cdf_iterator /= d_weights[ipoffset];
	    MCLS_CHECK( *cdf_iterator >= 0.0 );
	}
	MCLS_CHECK( 1.0 == d_cdfs[ipoffset].back() );
    }
}

//---------------------------------------------------------------------------//
/*
 * \brief Build boundary data.
 */
template<class Vector, class Matrix, class RNG>
void ForwardDomain<Vector,Matrix,RNG>::buildBoundary( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<const Matrix>& base_A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );

    // Get the next set of off-process rows. This is the boundary. If we
    // transition to these then we have left the local domain.
    Teuchos::RCP<Matrix> A_boundary = MT::copyNearestNeighbors( *A, 1 );

    // Get the boundary rows.
    Ordinal global_row = 0;
    Teuchos::Array<Ordinal> boundary_rows;
    for ( Ordinal i = 0; i < MT::getLocalNumRows( *A_boundary ); ++i )
    {
	global_row = MT::getGlobalRow( *A_boundary, i );
	if ( !isGlobalState(global_row) )
	{
	    boundary_rows.push_back( global_row );
	}
    }

    // Get the owning ranks for the boundary rows.
    Teuchos::Array<int> boundary_ranks( boundary_rows.size() );
    MT::getGlobalRowRanks( *base_A, boundary_rows(), boundary_ranks() );

    // Process the boundary data.
    Teuchos::Array<int>::const_iterator send_rank_it;
    Teuchos::Array<int>::const_iterator bnd_rank_it;
    typename Teuchos::Array<Ordinal>::const_iterator bnd_row_it;
    for ( bnd_row_it = boundary_rows.begin(), 
	 bnd_rank_it = boundary_ranks.begin();
	  bnd_row_it != boundary_rows.end();
	  ++bnd_row_it, ++bnd_rank_it )
    {
	MCLS_CHECK( *bnd_rank_it != -1 );

	// Look for the owning process in the send rank array.
	send_rank_it = std::find( d_send_ranks.begin(), 
				  d_send_ranks.end(),
				  *bnd_rank_it );

	// If it is new, add it to the send rank array.
	if ( send_rank_it == d_send_ranks.end() )
	{
	    d_send_ranks.push_back( *bnd_rank_it );
	    d_bnd_to_neighbor[*bnd_row_it] = d_send_ranks.size()-1;
	}

	// Otherwise, just add it to the boundary state to local id table.
	else
	{
	    d_bnd_to_neighbor[*bnd_row_it] =
		std::distance( 
		    Teuchos::as<Teuchos::Array<int>::const_iterator>(
			d_send_ranks.begin()), send_rank_it);
	}
    }

    MCLS_ENSURE( d_bnd_to_neighbor.size() == 
		 Teuchos::as<std::size_t>(boundary_rows.size()) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_FORWARDDOMAIN_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_ForwardDomain_impl.hpp
// ---------------------------------------------------------------------------//
