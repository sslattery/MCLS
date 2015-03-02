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

#include "MCLS_config.hpp"
#include "MCLS_VectorExport.hpp"

#include <Teuchos_as.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Tpetra_Distributor.hpp>
#include <Tpetra_Map.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

#include <AnasaziGeneralizedDavidsonSolMgr.hpp>
#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziTpetraAdapter.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Default constructor.
 */
template<class Vector, class Matrix, class RNG, class Tally>
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::AlmostOptimalDomain(
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::ParameterList& plist )
    : d_rng_dist( RDT::create(0.0, 1.0) )
    , d_history_length( 10 )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    MCLS_REQUIRE( Teuchos::nonnull(x) );

    // Build the domain data.
    buildDomain( A, plist );

    // Create the tally.
    d_tally = TT::create( x );

    // Compute the necessary and sufficient Monte Carlo convergence condition
    // if requested.
    if ( plist.isParameter("Compute Convergence Criteria") )
    {
	if ( plist.get<bool>("Compute Convergence Criteria") )
	{
	    Teuchos::Array<double> criteria = computeConvergenceCriteria();

	    if ( 0 == d_comm->getRank() )
	    {
		std::cout << std::endl;
		std::cout << "Neumann/Ulam Convergence Criteria" << std::endl;
		std::cout << "---------------------------------" << std::endl;
		std::cout << "rho(H)  = " << criteria[0] << std::endl;
		std::cout << "rho(H+) = " << criteria[1] << std::endl;
		std::cout << "rho(H*) = " << criteria[2] << std::endl;
		std::cout << "---------------------------------" << std::endl;
		std::cout << std::endl;
	    }
	}
    }

    MCLS_ENSURE( Teuchos::nonnull(d_tally) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Build the domain and return the global rows of the tally on this
 * process.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::buildDomain(
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::ParameterList& plist )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );

    // Get the history length.
    if ( plist.isParameter("History Length") )
    {
	d_history_length = plist.get<int>("History Length");
    }

    // Get the total number of local rows.
    int num_rows = MT::getLocalNumRows( *A );

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
    MCLS_CHECK( 0.0 < relaxation );
    addMatrixToDomain( A, relaxation );

    // Get the boundary states and their owning process ranks.
    buildBoundary( A );

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
    d_comm = MT::getComm(*A);
    Tpetra::Distributor distributor( d_comm );
    distributor.createFromSends( d_send_ranks() );
    d_receive_ranks = distributor.getImagesFrom();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank from which we will receive.
 */
template<class Vector, class Matrix, class RNG, class Tally>
int
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::receiveNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < d_receive_ranks.size() );
    return d_receive_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank to which we will send.
 */
template<class Vector, class Matrix, class RNG, class Tally>
int
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::sendNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < d_send_ranks.size() );
    return d_send_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain that owns a boundary state (local neighbor
 * id).
 */
template<class Vector, class Matrix, class RNG, class Tally>
int AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::owningNeighbor(
    const Ordinal& state ) const
{
    typename std::unordered_map<Ordinal,int>::const_iterator neighbor = 
	d_bnd_to_neighbor.find( state );
    MCLS_REQUIRE( neighbor != d_bnd_to_neighbor.end() );
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
/*
 * \brief Add matrix data to the local domain.
 */
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::addMatrixToDomain( 
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

    // Add row-by-row.
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

	// The final value in the non-normalized CDF is the absolute value of
	// the weight for this row. This is the absolute value row sum of the
	// iteration matrix.
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
template<class Vector, class Matrix, class RNG, class Tally>
void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::buildBoundary( 
    const Teuchos::RCP<const Matrix>& A )
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
    MT::getGlobalRowRanks( *A, boundary_rows(), boundary_ranks() );

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
// Compute the spectral radius of H and H*.
template<class Vector, class Matrix, class RNG, class Tally>
Teuchos::Array<double>
AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::computeConvergenceCriteria() const
{
    // Get the map for the operators.
    int num_rows = VT::getLocalLength( *TT::getVector(*d_tally) );
    Teuchos::RCP<const Tpetra::Map<int,Ordinal> > map =
	TT::getVector(*d_tally)->getMap();
    Teuchos::ArrayView<const Ordinal> local_rows =
	map->getNodeElementList();

    // Find the convergence criteria.
    Teuchos::Array<double> evals(3);

    // Spectral radius of H.
    double max_entries = 0;
    {
	Teuchos::RCP<Tpetra::CrsMatrix<double,int,Ordinal> > H =
	    Tpetra::createCrsMatrix<double,int,Ordinal>( map );
	for ( int i = 0; i < num_rows; ++i )
	{
	    H->insertGlobalValues(
		local_rows[i],
		(*d_global_columns[i])(),
		d_h[i]() );
	}
	H->fillComplete();
	max_entries = H->getGlobalMaxNumRowEntries();
	evals[0] = computeSpectralRadius( H );
    }

    // Allocate a work array.
    Teuchos::Array<double> values( max_entries );
    int row_size = 0;

    // Spectral radius of H+.
    {
	Teuchos::RCP<Tpetra::CrsMatrix<double,int,Ordinal> > H_plus =
	    Tpetra::createCrsMatrix<double,int,Ordinal>( map );
	double max_row_sum = 0.0;
	double min_row_sum = std::numeric_limits<double>::max();
	double row_sum = 0.0;
	for ( int i = 0; i < num_rows; ++i )
	{
	    row_sum = 0.0;
	    row_size = d_global_columns[i]->size();
	    for ( int j = 0; j < row_size; ++j )
	    {
		values[j] = std::abs(d_h[i][j]);
		row_sum += values[j];
	    }

	    H_plus->insertGlobalValues(
		local_rows[i],
		(*d_global_columns[i])(),
		values(0,row_size) );

	    max_row_sum = std::max( max_row_sum, row_sum );
	    min_row_sum = std::min( min_row_sum, row_sum );
	}
	H_plus->fillComplete();
	evals[1] = computeSpectralRadius( H_plus );

	double global_max_row_sum = 0.0;
	Teuchos::reduceAll( *d_comm, Teuchos::REDUCE_MAX, 
			    max_row_sum, Teuchos::ptr(&global_max_row_sum) );
	double global_min_row_sum = 0.0;
	Teuchos::reduceAll( *d_comm, Teuchos::REDUCE_MIN, 
			    min_row_sum, Teuchos::ptr(&global_min_row_sum) );

	// Balance Criteria.
	Teuchos::RCP<Tpetra::CrsMatrix<double,int,Ordinal> > H_T = 
	    MatrixTraits<Tpetra::Vector<double,int,Ordinal>,
			 Tpetra::CrsMatrix<double,int,Ordinal> >::copyTranspose(*H_plus);
	Teuchos::ArrayView<const int> h_indices;
	Teuchos::ArrayView<const int> h_T_indices;
	Teuchos::ArrayView<const double> h_values;
	Teuchos::ArrayView<const double> h_T_values;
	double row_norm = 0.0;
	double col_norm = 0.0;
	double norm_ratio = 0.0;
	double max_val = 0.0;
	double min_val = std::numeric_limits<double>::max();
	double ave_val = 0.0;
	for ( int i = 0; i < num_rows; ++i )
	{
	    H_plus->getLocalRowView( i, h_indices, h_values );
	    H_T->getLocalRowView( i, h_T_indices, h_T_values );
	    row_norm = *std::max_element( h_values.begin(), h_values.end() );
	    col_norm = *std::max_element( h_T_values.begin(), h_T_values.end() );
	    norm_ratio = row_norm / col_norm;
	    min_val = std::min( min_val, norm_ratio );
	    max_val = std::max( max_val, norm_ratio );
	    ave_val += norm_ratio;
	}
	ave_val /= num_rows;

	double global_min = 0.0;
	double global_max = 0.0;
	double global_ave = 0.0;
	Teuchos::reduceAll( *d_comm, Teuchos::REDUCE_MIN, 
			    min_val, Teuchos::ptr(&global_min) );
	Teuchos::reduceAll( *d_comm, Teuchos::REDUCE_MAX, 
			    max_val, Teuchos::ptr(&global_max) );
	Teuchos::reduceAll( *d_comm, Teuchos::REDUCE_SUM, 
			    ave_val, Teuchos::ptr(&global_ave) );
	global_ave /= d_comm->getSize();

	if ( 0 == d_comm->getRank() )
	{
	    std::cout << std::endl;
	    std::cout << "||H||_inf = " << global_max_row_sum << std::endl;
	    std::cout << "sum|H|_min = " << global_min_row_sum << std::endl;
	    std::cout << std::endl;
	    std::cout << "H Balance Parameters" << std::endl;
	    std::cout << "min: " << global_min << std::endl;
	    std::cout << "max: " << global_max << std::endl;
	    std::cout << "ave: " << global_ave << std::endl;
	}
    }

    // Spectral radius of H*.
    {
	Teuchos::RCP<Tpetra::CrsMatrix<double,int,Ordinal> > H_star =
	    Tpetra::createCrsMatrix<double,int,Ordinal>( map );
	int h_sign = 0;
	for ( int i = 0; i < num_rows; ++i )
	{
	    row_size = d_global_columns[i]->size();
	    for ( int j = 0; j < row_size; ++j )
	    {
		h_sign = ( d_h[i][j] > 0.0 ) ? 1 : -1;
		values[j] = d_h[i][j] * h_sign * d_weights[i];
	    }

	    H_star->insertGlobalValues(
		local_rows[i],
		(*d_global_columns[i])(),
		values(0,row_size) );
	}
	H_star->fillComplete();
	evals[2] = computeSpectralRadius( H_star );
    }

    return evals;
}

//---------------------------------------------------------------------------//
// Given a crs matrix, compute its spectral radius.
template<class Vector, class Matrix, class RNG, class Tally>
double AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::computeSpectralRadius( 
    const Teuchos::RCP<Tpetra::CrsMatrix<double,int,Ordinal> >& matrix ) const
{
    typedef double Scalar;
    typedef Tpetra::Operator<Scalar,int,Ordinal> OP;
    typedef Tpetra::MultiVector<Scalar,int,Ordinal> MV;

    int nev = 1;
    int block_size = 1;
    int max_dim = 40;
    int max_restarts = 100;
    double tol = 1.0e-6;

    int verbosity = Anasazi::Errors + Anasazi::Warnings;
#if HAVE_MCLS_DBC
    verbosity += Anasazi::FinalSummary + Anasazi::TimingDetails;
#endif

    Teuchos::ParameterList parameters;
    parameters.set( "Verbosity", verbosity );
    parameters.set( "Which", "LM" );
    parameters.set( "Block Size", block_size );
    parameters.set( "Maximum Subspace Dimension", max_dim );
    parameters.set( "Maximum Restarts", max_restarts );
    parameters.set( "Convergence Tolerance", tol );
    parameters.set( "Initial Guess", "User" );

    Teuchos::RCP<MV> evec = Tpetra::createMultiVector<Scalar,int,Ordinal>( 
	matrix->getMap(), block_size );
    evec->randomize();

    Teuchos::RCP<Anasazi::BasicEigenproblem<Scalar,MV,OP> > eigenproblem =
	Teuchos::rcp( new Anasazi::BasicEigenproblem<Scalar,MV,OP>() );
    eigenproblem->setA( matrix );
    eigenproblem->setInitVec( evec );
    eigenproblem->setNEV( nev );
    eigenproblem->setProblem();

    Anasazi::GeneralizedDavidsonSolMgr<Scalar,MV,OP> eigensolver( 
	eigenproblem, parameters );
    MCLS_CHECK_ERROR_CODE( eigensolver.solve() );
    Anasazi::Eigensolution<Scalar,MV> sol = eigenproblem->getSolution();
    std::vector<Anasazi::Value<Scalar> > evals = sol.Evals;
    MCLS_ENSURE( 1 == evals.size() );
    double eval_mag = std::sqrt(evals[0].realpart*evals[0].realpart +
				evals[0].imagpart*evals[0].imagpart);

    return eval_mag;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_ALMOSTOPTIMALDOMAIN_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_AlmostOptimalDomain_impl.hpp
// ---------------------------------------------------------------------------//
