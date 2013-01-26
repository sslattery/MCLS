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
 * \file MCLS_AdjointDomain_impl.hpp
 * \author Stuart R. Slattery
 * \brief AdjointDomain implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTDOMAIN_IMPL_HPP
#define MCLS_ADJOINTDOMAIN_IMPL_HPP

#include <algorithm>

#include <MCLS_MatrixTraits.hpp>
#include <MCLS_Serializer.hpp>

#include <Teuchos_as.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Distributor.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Matrix constructor.
 */
template<class Vector, class Matrix>
AdjointDomain<Vector,Matrix>::AdjointDomain( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::ParameterList& plist )
{
    Require( !A.is_null() );
    Require( !x.is_null() );

    // Generate the transpose of the operator.
    Teuchos::RCP<Matrix> A_T = MT::copyTranspose( *A );

    // Generate the overlap for the transpose operator.
    int num_overlap = plist.get<int>( "Overlap Size" );
    Require( num_overlap >= 0 );
    Teuchos::RCP<Matrix> A_T_overlap = 
	MT::copyNearestNeighbors( *A_T, num_overlap );

    // Generate a solution vector with the overlap decomposition.
    Teuchos::RCP<Vector> x_overlap = 
	MT::cloneVectorFromMatrixRows( *A_T_overlap );

    // Build the adjoint tally from the solution vector and the overlap.
    d_tally = Teuchos::rcp( new TallyType( x, x_overlap ) );

    // Allocate space in local row data arrays.
    int num_rows = 
	MT::getLocalNumRows( *A_T ) + MT::getLocalNumRows( *A_T_overlap );
    d_columns.resize( num_rows );
    d_cdfs.resize( num_rows );
    d_weights.resize( num_rows );

    // Build the local CDFs and weights.
    addMatrixToDomain( A_T );
    addMatrixToDomain( A_T_overlap );

    // Get the boundary states and their owning process ranks.
    if ( num_overlap == 0 )
    {
	buildBoundary( A_T, A );
    }
    else
    {
	buildBoundary( A_T_overlap, A );
    }

    // By building the boundary data, now we know where we are sending
    // data. Find out who we are receiving from.
    Tpetra::Distributor distributor( MT::getComm(*A) );
    distributor.createFromSends( d_send_ranks() );
    d_receive_ranks = distributor.getImagesFrom();

    Ensure( !d_tally.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Deserializer constructor.
 */
template<class Vector, class Matrix>
AdjointDomain<Vector,Matrix>::AdjointDomain( 
    const Teuchos::ArrayView<char>& buffer )
{
    Require( Teuchos::as<std::size_t>(buffer.size()) == d_packed_bytes );

    Deserializer ds;
    ds.set_buffer( buffer() );

    // Unpack of the local row indexer.

    // Unpack up the local columns.

    // Unpack up the local cdfs.

    // Unpack up the local weights.

    // Unpack up the receive ranks.

    // Unpack up the send ranks.

    // Unpack up the boundary-to-neighbor id table.

    // Unpack up the tally base rows.

    // Unpack up the tally overlap rows.

    // Build the tally.

}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the domain into a buffer.
 */
template<class Vector, class Matrix>
Teuchos::Array<char> AdjointDomain<Vector,Matrix>::pack()
{
    Require( d_packed_bytes );

    Teuchos::Array<char> buffer( d_packed_bytes );
    Serializer s;
    s.setBuffer( buffer() );

    // Pack of the local row indexer.

    // Pack up the local columns.

    // Pack up the local cdfs.

    // Pack up the local weights.

    // Pack up the receive ranks.

    // Pack up the send ranks.

    // Pack up the boundary-to-neighbor id table.

    // Pack up the tally base rows.

    // Pack up the tally overlap rows.
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank from which we will receive.
 */
template<class Vector, class Matrix>
int AdjointDomain<Vector,Matrix>::receiveNeighborRank( int n ) const
{
    Require( n >= 0 && n < d_receive_ranks.size() );
    return d_receive_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank to which we will send.
 */
template<class Vector, class Matrix>
int AdjointDomain<Vector,Matrix>::sendNeighborRank( int n ) const
{
    Require( n >= 0 && n < d_send_ranks.size() );
    return d_send_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain that owns a boundary state (local neighbor
 * id).
 */
template<class Vector, class Matrix>
int AdjointDomain<Vector,Matrix>::owningNeighbor( const Ordinal& state )
{
    typename std::tr1::unordered_map<Ordinal,int>::const_iterator neighbor =
	d_bnd_to_neighbor.find( state );
    Require( neighbor != d_bnd_to_neighbor.end() );
    return neighbor->second;
}

//---------------------------------------------------------------------------//
/*
 * \brief Add matrix data to the local domain.
 */
template<class Vector, class Matrix>
void AdjointDomain<Vector,Matrix>::addMatrixToDomain( 
    const Teuchos::RCP<const Matrix>& A )
{
    Require( !A.is_null() );

    Ordinal local_num_rows = MT::getLocalNumRows( *A );
    Ordinal global_row = 0;
    int offset = d_row_indexer.size();
    int max_entries = MT::getGlobalMaxNumRowEntries( *A );
    std::size_t num_entries = 0;
    typename Teuchos::Array<Ordinal>::const_iterator diagonal_iterator;
    Teuchos::Array<double>::iterator cdf_iterator;

    for ( Ordinal i = 0; i < local_num_rows; ++i )
    {
	// Add the global row id and local row id to the indexer.
	global_row = MT::getGlobalRow(*A, i);
	d_row_indexer[global_row] = i+offset;

	// Allocate column and CDF memory for this row.
	d_columns[i+offset].resize( max_entries );
	d_cdfs[i+offset].resize( max_entries );

	// Add the columns and base PDF values for this row.
	MT::getGlobalRowCopy( *A, 
			      global_row,
			      d_columns[i+offset](), 
			      d_cdfs[i+offset](), 
			      num_entries );

	// Check for degeneracy.
	Check( num_entries > 0 );

	// Resize local column and CDF arrays for this row.
	d_columns[i+offset].resize( num_entries );
	d_cdfs[i+offset].resize( num_entries );

	// If this row contains an entry on the diagonal, subtract 1 for the
	// identity matrix (H^T = I-A^T).
	diagonal_iterator = std::find( d_columns[i+offset].begin(),
				       d_columns[i+offset].end(),
				       global_row );
	if ( diagonal_iterator != d_columns[i+offset].end() )
	{
	    d_cdfs[i+offset][ std::distance(
		    Teuchos::as<typename Teuchos::Array<Ordinal>::const_iterator>(
			d_columns[i+offset].begin()), diagonal_iterator) ] -= 1;
	}

	// Accumulate the absolute value of the PDF values to get a
	// non-normalized CDF for the row.
	d_cdfs[i+offset].front() = std::abs( d_cdfs[i+offset].front() );
	for ( cdf_iterator = d_cdfs[i+offset].begin()+1;
	      cdf_iterator != d_cdfs[i+offset].end();
	      ++cdf_iterator )
	{
	    *cdf_iterator = std::abs( *cdf_iterator ) + *(cdf_iterator-1);
	}

	// The final value in the non-normalized CDF is the weight for this
	// row. This is the absolute value row sum of the iteration matrix.
	d_weights[i+offset] = d_cdfs[i+offset].back();
	Check( d_weights[i+offset] >= 0 );

	// Normalize the CDF for the row.
	for ( cdf_iterator = d_cdfs[i+offset].begin();
	      cdf_iterator != d_cdfs[i+offset].end();
	      ++cdf_iterator )
	{
	    *cdf_iterator /= d_weights[i+offset];
	    Check( *cdf_iterator >= 0.0 );
	}

	Check( std::abs(1.0 - d_cdfs[i+offset].back()) < 1.0e-6 );
    }
}

//---------------------------------------------------------------------------//
/*
 * \brief Build boundary data.
 */
template<class Vector, class Matrix>
void AdjointDomain<Vector,Matrix>::buildBoundary( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<const Matrix>& base_A )
{
    Require( !A.is_null() );

    // Get the next set of off-process rows. This is the boundary. If we
    // transition to these then we have left the local domain.
    Teuchos::RCP<Matrix> A_boundary = MT::copyNearestNeighbors( *A, 1 );

    // Get the boundary rows.
    Ordinal global_row = 0;
    Teuchos::Array<Ordinal> boundary_rows;
    for ( Ordinal i = 0; i < MT::getLocalNumRows( *A_boundary ); ++i )
    {
	global_row = MT::getGlobalRow( *A_boundary, i );
	if ( !isLocalState(global_row) )
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
	Check( *bnd_rank_it != -1 );

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

    Ensure( d_bnd_to_neighbor.size() == 
	    Teuchos::as<std::size_t>(boundary_rows.size()) );
}

//---------------------------------------------------------------------------//
// Static members.
//---------------------------------------------------------------------------//
template<class Vector, class Matrix>
std::size_t Domain<Vector,Matrix>::d_packed_bytes = 0;

//---------------------------------------------------------------------------//
/*!
 * \brief Set the byte size of the packed domain state.
 */
template<class Vector, class Matrix>
void Domain<Vector,Matrix>::setByteSize( Ordinal num_rows, Ordinal max_col, 
					 int num_receive, int num_send, 
					 Ordinal num_boundary,
					 int num_base, int num_overlap )
{
    d_packed_bytes = 
	num_rows*( 
	sizeof(Ordinal) + sizeof(int) + sizeof(Ordinal)*(max_col+1) +
	sizeof(double)*(max_col+1) + sizeof(double) + sizeof(int) +
	sizeof(int) + sizeof(Ordinal) + sizeof(int) 
	)
	+ (num_base+num_overlap)*sizeof(Ordinal);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number of bytes in the packed domain state.
 */
template<class Vector, class Matrix>
std::size_t Domain<Vector,Matrix>::getPackedBytes()
{
    Require( d_packed_bytes );
    return d_packed_bytes;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_ADJOINTDOMAIN_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointDomain_impl.hpp
// ---------------------------------------------------------------------------//

