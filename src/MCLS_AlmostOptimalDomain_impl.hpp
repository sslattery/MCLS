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
 * \file MCLS_AlmostOptimalDomain_impl.hpp
 * \author Stuart R. Slattery
 * \brief AlmostOptimalDomain implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ALMOSTOPTIMALDOMAIN_IMPL_HPP
#define MCLS_ALMOSTOPTIMALDOMAIN_IMPL_HPP

#include <algorithm>
#include <limits>
#include <string>

#include "MCLS_Estimators.hpp"
#include "MCLS_VectorExport.hpp"

#include <Teuchos_as.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Distributor.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Default constructor.
 */
template<class Vector, class Matrix, class RNG, class Tally>
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::AlmostOptimalDomain()
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Build the domain and return the global rows of the tally on this
 * process including overlap.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::buildDomain(
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::ParameterList& plist,
    Teuchos::Array<Ordinal>& local_tally_states )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    MCLS_REQUIRE( Teuchos::nonnull(x) );

    // Get the estimator type. User the collision estimator as the default.
    b_estimator = Estimator::COLLISION;
    if ( plist.isParameter("Estimator Type") )
    {
	if ( "Collision" == plist.get<std::string>("Estimator Type") )
	{
	    b_estimator = Estimator::COLLISION;
	}
	else if ( "Expected Value" == plist.get<std::string>("Estimator Type") )
	{
	    b_estimator = Estimator::EXPECTED_VALUE;
	}
    }

    // Get the amount of overlap.
    int num_overlap = plist.get<int>( "Overlap Size" );
    MCLS_REQUIRE( num_overlap >= 0 );

    // Set of local tally states.
    std::set<Ordinal> tally_states;

    // Generate the Monte Carlo domain.
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
        b_rng_dists = Teuchos::ArrayRCP<Teuchos::RCP<RandomDistribution> >( num_rows );
        b_global_columns = 
            Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > >( num_rows );
        b_local_columns = 
            Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<int> > >( num_rows );
        b_h = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );

        // Build the iteration matrix.
	double relaxation = 1.0;
	if ( plist.isParameter("Neumann Relaxation") )
	{
	    relaxation = plist.get<double>("Neumann Relaxation");
	}
	MCLS_CHECK( 0.0 < relaxation );
        addMatrixToDomain( A, tally_states, relaxation );
        if ( num_overlap > 0 )
        {
            addMatrixToDomain( A_overlap, tally_states, relaxation );
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

    // Make the set of local columns and row distributions. If the local
    // column is not a global row then make it invalid to indicate that we
    // have left the domain.
    typename Teuchos::ArrayRCP<Teuchos::RCP<RandomDistribution> >::iterator
	rng_dist_it;
    typename Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator global_it;
    typename Teuchos::Array<Ordinal>::const_iterator gcol_it;
    Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<int> > >::iterator local_it;
    Teuchos::Array<int>::iterator lcol_it;
    for ( global_it = b_global_columns.begin(),
	   local_it = b_local_columns.begin(),
	rng_dist_it = b_rng_dists.begin();
	  global_it != b_global_columns.end();
	  ++global_it, ++local_it, ++rng_dist_it )
    {
	*rng_dist_it = RDT::create( 0, (*global_it)->size()-1 );
	*local_it = Teuchos::rcp(
	    new Teuchos::Array<int>((*global_it)->size()) );
	for ( gcol_it = (*global_it)->begin(),
	      lcol_it = (*local_it)->begin();
	      gcol_it != (*global_it)->end();
	      ++gcol_it, ++lcol_it )
	{
	    if ( b_g2l_row_indexer.count(*gcol_it) )
	    {
		*lcol_it = b_g2l_row_indexer.find( *gcol_it )->second;
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
    distributor.createFromSends( b_send_ranks() );
    b_receive_ranks = distributor.getImagesFrom();

    // Create the tally vector.
    local_tally_states.resize( tally_states.size() );
    std::copy( tally_states.begin(), tally_states.end(),
               local_tally_states.begin() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the domain into a buffer.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::packDomain( Serializer& s ) const
{
    // Pack the estimator type.
    s << Teuchos::as<int>(b_estimator);

    // Pack the local number of rows.
    s << Teuchos::as<Ordinal>(b_g2l_row_indexer.size());

    // Pack in the number of receive neighbors.
    s << Teuchos::as<int>(b_receive_ranks.size());

    // Pack in the number of send neighbors.
    s << Teuchos::as<int>(b_send_ranks.size());

    // Pack in the number of boundary states.
    s << Teuchos::as<Ordinal>(b_bnd_to_neighbor.size());

    // Pack in the number of base rows in the tally.
    s << Teuchos::as<Ordinal>(b_tally->numBaseRows());

    // Pack up the global-to-local row indexer by key-value pairs.
    typename std::unordered_map<Ordinal,int>::const_iterator row_index_it;
    for ( row_index_it = b_g2l_row_indexer.begin();
	  row_index_it != b_g2l_row_indexer.end();
	  ++row_index_it )
    {
	s << row_index_it->first << row_index_it->second;
    }

    // Pack up the local columns.
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator column_it;
    typename Teuchos::Array<Ordinal>::const_iterator index_it;
    for( column_it = b_global_columns.begin(); 
	 column_it != b_global_columns.end(); 
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

    // Pack the iteration matrix values.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::const_iterator h_it;
    Teuchos::ArrayRCP<double>::const_iterator h_val_it;
    for( h_it = b_h.begin(); h_it != b_h.end(); ++h_it )
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

    // Pack up the receive ranks.
    Teuchos::Array<int>::const_iterator receive_it;
    for ( receive_it = b_receive_ranks.begin();
	  receive_it != b_receive_ranks.end();
	  ++receive_it )
    {
	s << *receive_it;
    }

    // Pack up the send ranks.
    Teuchos::Array<int>::const_iterator send_it;
    for ( send_it = b_send_ranks.begin();
	  send_it != b_send_ranks.end();
	  ++send_it )
    {
	s << *send_it;
    }

    // Pack up the boundary-to-neighbor id table.
    typename std::unordered_map<Ordinal,int>::const_iterator bnd_it;
    for ( bnd_it = b_bnd_to_neighbor.begin();
	  bnd_it != b_bnd_to_neighbor.end();
	  ++bnd_it )
    {
	s << bnd_it->first << bnd_it->second;
    }

    // Pack up the tally base rows.
    Teuchos::Array<Ordinal> base_rows = b_tally->baseRows();
    typename Teuchos::Array<Ordinal>::const_iterator base_it;
    for ( base_it = base_rows.begin();
	  base_it != base_rows.end();
	  ++base_it )
    {
	s << *base_it;
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Unpack the domain from a buffer.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::unpackDomain( 
    Deserializer& ds, Teuchos::Array<Ordinal>& base_rows )
{
    Ordinal num_rows = 0;
    int num_receives = 0;
    int num_sends = 0;
    Ordinal num_bnd = 0;
    Ordinal num_base = 0;

    // Unpack the estimator type.
    ds >> b_estimator;
    MCLS_CHECK( b_estimator >= 0 );

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

    // Unpack the global-to-local and local-to-global row indexers by
    // key-value pairs.
    Ordinal global_row = 0;
    int local_row = 0;
    for ( Ordinal n = 0; n < num_rows; ++n )
    {
	ds >> global_row >> local_row;
	b_g2l_row_indexer[global_row] = local_row;
    }

    // Unpack the local columns.
    std::set<Ordinal> im_unique_cols;
    b_global_columns = 
        Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > >( num_rows );
    Ordinal num_cols = 0;
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::iterator column_it;
    typename Teuchos::Array<Ordinal>::iterator index_it;
    for( column_it = b_global_columns.begin(); 
	 column_it != b_global_columns.end(); 
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

    // Unpack the iteration matrix values.
    b_h = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );
    int num_values = 0;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::iterator h_it;
    Teuchos::ArrayRCP<double>::iterator h_val_it;
    for( h_it = b_h.begin(); h_it != b_h.end(); ++h_it )
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

    // Unpack the receive ranks.
    b_receive_ranks.resize( num_receives );
    Teuchos::Array<int>::iterator receive_it;
    for ( receive_it = b_receive_ranks.begin();
	  receive_it != b_receive_ranks.end();
	  ++receive_it )
    {
	ds >> *receive_it;
    }

    // Unpack the send ranks.
    b_send_ranks.resize( num_sends );
    Teuchos::Array<int>::iterator send_it;
    for ( send_it = b_send_ranks.begin();
	  send_it != b_send_ranks.end();
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
	b_bnd_to_neighbor[boundary_row] = neighbor;
    }

    // Unpack the tally base rows.
    base_rows.resize( num_base );
    typename Teuchos::Array<Ordinal>::iterator base_it;
    for ( base_it = base_rows.begin();
	  base_it != base_rows.end();
	  ++base_it )
    {
	ds >> *base_it;
    }

    // Make the set of local columns and row distributions. If the local
    // column is not a global row then make it invalid to indicate that we
    // have left the domain.
    b_rng_dists = Teuchos::ArrayRCP<Teuchos::RCP<RandomDistribution> >( num_rows );
    typename Teuchos::ArrayRCP<Teuchos::RCP<RandomDistribution> >::iterator
	rng_dist_it;
    b_local_columns = 
        Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<int> > >( num_rows );
    typename Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator global_it;
    typename Teuchos::Array<Ordinal>::const_iterator gcol_it;
    Teuchos::ArrayRCP<
	Teuchos::RCP<Teuchos::Array<int> > >::iterator local_it;
    Teuchos::Array<int>::iterator lcol_it;
    for ( global_it = b_global_columns.begin(),
	   local_it = b_local_columns.begin(),
	rng_dist_it = b_rng_dists.begin();
	  global_it != b_global_columns.end();
	  ++global_it, ++local_it, ++rng_dist_it )
    {
	*rng_dist_it = RDT::create( 0, (*global_it)->size()-1 );
	*local_it = Teuchos::rcp(
	    new Teuchos::Array<int>((*global_it)->size()) );
	for ( gcol_it = (*global_it)->begin(),
	      lcol_it = (*local_it)->begin();
	      gcol_it != (*global_it)->end();
	      ++gcol_it, ++lcol_it )
	{
	    if ( b_g2l_row_indexer.count(*gcol_it) )
	    {
		*lcol_it = b_g2l_row_indexer.find( *gcol_it )->second;
	    }
	    else
	    {
		*lcol_it = Teuchos::OrdinalTraits<int>::invalid();
	    }
	}
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank from which we will receive.
 */
template<class Vector, class Matrix, class RNG, class Tally>
int AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::receiveNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < b_receive_ranks.size() );
    return b_receive_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank to which we will send.
 */
template<class Vector, class Matrix, class RNG, class Tally>
int AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::sendNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < b_send_ranks.size() );
    return b_send_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain that owns a boundary state (local neighbor
 * id).
 */
template<class Vector, class Matrix, class RNG, class Tally>
int AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::owningNeighbor( const Ordinal& state ) const
{
    typename std::unordered_map<Ordinal,int>::const_iterator neighbor = 
	b_bnd_to_neighbor.find( state );
    MCLS_REQUIRE( neighbor != b_bnd_to_neighbor.end() );
    return neighbor->second;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the local states owned by this domain.
 */
template<class Vector, class Matrix, class RNG, class Tally>
Teuchos::Array<typename AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::Ordinal>
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::localStates() const
{
    Teuchos::Array<Ordinal> states( b_g2l_row_indexer.size() );
    typename Teuchos::Array<Ordinal>::iterator state_it;
    typename std::unordered_map<Ordinal,int>::const_iterator map_it;
    for ( map_it = b_g2l_row_indexer.begin(), state_it = states.begin();
          map_it != b_g2l_row_indexer.end();
          ++map_it, ++state_it )
    {
        *state_it = map_it->first;
    }

    return states;
}

//---------------------------------------------------------------------------//
/*
 * \brief Add matrix data to the local domain.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::addMatrixToDomain( 
    const Teuchos::RCP<const Matrix>& A,
    std::set<Ordinal>& tally_states,
    const double relaxation )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );

    Ordinal local_num_rows = MT::getLocalNumRows( *A );
    Ordinal global_row = 0;
    int offset = b_g2l_row_indexer.size();
    int ipoffset = 0;
    int max_entries = MT::getGlobalMaxNumRowEntries( *A );
    std::size_t num_entries = 0;
    Teuchos::Array<double> values( max_entries );

    // Add row-by-row.
    for ( Ordinal i = 0; i < local_num_rows; ++i )
    {
	// Get the offset row index.
	ipoffset = i+offset;

	// Add the global row id and local row id to the indexer.
	global_row = MT::getGlobalRow(*A, i);
	b_g2l_row_indexer[global_row] = ipoffset;

	// Allocate column and value memory for this row.
        b_global_columns[ipoffset] = 
            Teuchos::rcp( new Teuchos::Array<Ordinal>(max_entries) );
	values.resize( max_entries );

	// Add the columns and base PDF values for this row.
	MT::getGlobalRowCopy( *A, 
			      global_row,
			      (*b_global_columns[ipoffset])(), 
			      values(),
			      num_entries );

	// Check for degeneracy.
	MCLS_CHECK( num_entries > 0 );

	// Resize local column array for this row.
	b_global_columns[ipoffset]->resize( num_entries );

	// Create the iteration matrix.
	for ( std::size_t j = 0; j < num_entries; ++j )
	{
	    // Subtract the operator from the identity matrix.
	    values[j] = ( (*b_global_columns[ipoffset])[j] == global_row ) ?
		1.0 - relaxation*values[j] : -relaxation*values[j];

	    // Mark any zero entries.
	    if ( std::abs(values[j]) < 
		 std::numeric_limits<double>::epsilon() )
	    {
		values[j] = std::numeric_limits<double>::max();
		(*b_global_columns[ipoffset])[j] = 
		    Teuchos::OrdinalTraits<Ordinal>::invalid();
	    }
	}

	// Extract any zero entries from the iteration matrix.
	Teuchos::Array<double>::iterator value_remove_it;
	value_remove_it = std::remove( 
	    values.begin(), values.end(), std::numeric_limits<double>::max() );
	values.resize( std::distance(values.begin(),value_remove_it) );

	typename Teuchos::Array<Ordinal>::iterator col_remove_it;
	col_remove_it = std::remove( b_global_columns[ipoffset]->begin(), 
				     b_global_columns[ipoffset]->end(),
				     Teuchos::OrdinalTraits<Ordinal>::invalid() );
	b_global_columns[ipoffset]->resize( 
	    std::distance(b_global_columns[ipoffset]->begin(), col_remove_it) );

        // Save the current values as the iteration matrix.
        b_h[ipoffset] = Teuchos::ArrayRCP<double>( values.size() );
        std::copy( values.begin(), values.end(), b_h[ipoffset].begin() );

        // If we're using the collision estimator, add the global row as a
        // local tally state.
        if ( Estimator::COLLISION == b_estimator )
        {
            tally_states.insert( global_row );
        }
        // Else if we're using the expected value estimator add the columns from
        // this row as local tally states.
        else if ( Estimator::EXPECTED_VALUE == b_estimator )
        {
            typename Teuchos::Array<Ordinal>::const_iterator col_it;
            for ( col_it = b_global_columns[ipoffset]->begin();
                  col_it != b_global_columns[ipoffset]->end()-1;
                  ++col_it )
            {
                tally_states.insert( *col_it );
            }
        }
        else
        {
            MCLS_INSIST( Estimator::COLLISION == b_estimator || 
                         Estimator::EXPECTED_VALUE == b_estimator,
                         "Unsupported estimator type" );
        }
    }
}

//---------------------------------------------------------------------------//
/*
 * \brief Build boundary data.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::buildBoundary( 
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
	send_rank_it = std::find( b_send_ranks.begin(), 
				  b_send_ranks.end(),
				  *bnd_rank_it );

	// If it is new, add it to the send rank array.
	if ( send_rank_it == b_send_ranks.end() )
	{
	    b_send_ranks.push_back( *bnd_rank_it );
	    b_bnd_to_neighbor[*bnd_row_it] = b_send_ranks.size()-1;
	}

	// Otherwise, just add it to the boundary state to local id table.
	else
	{
	    b_bnd_to_neighbor[*bnd_row_it] =
		std::distance( 
		    Teuchos::as<Teuchos::Array<int>::const_iterator>(
			b_send_ranks.begin()), send_rank_it);
	}
    }

    MCLS_ENSURE( b_bnd_to_neighbor.size() == 
		 Teuchos::as<std::size_t>(boundary_rows.size()) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_ALMOSTOPTIMALDOMAIN_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AlmostOptimalDomain_impl.hpp
// ---------------------------------------------------------------------------//
