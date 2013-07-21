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
 * \file MCLS_MaximalEntropyDomain_impl.hpp
 * \author Stuart R. Slattery
 * \brief MaximalEntropyDomain implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MAXIMALENTROPYDOMAIN_IMPL_HPP
#define MCLS_MAXIMALENTROPYDOMAIN_IMPL_HPP

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "MCLS_Serializer.hpp"
#include "MCLS_Estimators.hpp"
#include "MCLS_VectorExport.hpp"

#include <Teuchos_as.hpp>
#include <Teuchos_Array.hpp>

#include <Tpetra_Distributor.hpp>

#include <AnasaziConfigDefs.hpp>
#include <AnasaziTypes.hpp>
#include <AnasaziBasicEigenproblem.hpp>
#include <AnasaziBlockKrylovSchurSolMgr.hpp>
#include <AnasaziBasicOutputManager.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Matrix constructor.
 */
template<class Vector, class Matrix>
MaximalEntropyDomain<Vector,Matrix>::MaximalEntropyDomain( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<Vector>& x,
    const Teuchos::ParameterList& plist )
    : d_row_indexer( Teuchos::rcp(new MapType()) )
{
    MCLS_REQUIRE( !A.is_null() );
    MCLS_REQUIRE( !x.is_null() );

    // Get the estimator type. User the collision estimator as the default.
    d_estimator = Estimator::COLLISION;
    if ( plist.isParameter("Estimator Type") )
    {
	if ( "Collision" == plist.get<std::string>("Estimator Type") )
	{
	    d_estimator = Estimator::COLLISION;
	}
	else if ( "Expected Value" == plist.get<std::string>("Estimator Type") )
	{
	    d_estimator = Estimator::EXPECTED_VALUE;
	}
    }

    // Get the absorption probability. Default to 0.0.
    double abs_probability = 0.0;
    if ( plist.isParameter("Absorption Probability") )
    {
        abs_probability = plist.get<double>("Absorption Probability");
    }
    MCLS_CHECK( 0.0 <= abs_probability && abs_probability < 1.0 );

    // Get the amount of overlap.
    int num_overlap = plist.get<int>( "Overlap Size" );
    MCLS_REQUIRE( num_overlap >= 0 );

    // Set of local tally states.
    std::set<Ordinal> tally_states;

    // Build the reduced domain.
    Teuchos::RCP<Matrix> reduced_H;
    Teuchos::RCP<Vector> recovered_weights;
    {
        // Get the filter tolerance.
        double filter_tol = 0.0;
        if ( plist.isParameter("Domain Filter Tolerance") )
        {
            filter_tol = plist.get<double>("Domain Filter Tolerance");
        }
        MCLS_CHECK( 0.0 <= filter_tol );

        // Get the fill value.
        int fill_value = std::numeric_limits<int>::max();
        if ( plist.isParameter("Domain Fill Value") )
        {
            fill_value = plist.get<int>("Domain Fill Value");
        }
        MCLS_CHECK( 0 < fill_value );

        // Get the weight recovery value.
        double weight_recovery = 0.0;
        if ( plist.isParameter("Domain Weight Recovery") )
        {
            weight_recovery = plist.get<double>("Domain Weight Recovery");
        }
        MCLS_CHECK( 0.0 <= weight_recovery && 1.0 >= weight_recovery );
    
        // Get the Neumann relaxation parameter.
        double neumann_relax = 1.0;
        if ( plist.isParameter("Neumann Relaxation") )
        {
            neumann_relax = plist.get<double>("Neumann Relaxation");
        }
        MCLS_CHECK( 0.0 < neumann_relax );

        // Apply the reduced domain approximation to build a reduced iteration
        // matrix. 
        MA::reducedDomainApproximation( *A, neumann_relax, filter_tol, 
                                        fill_value, weight_recovery, 
                                        reduced_H, recovered_weights );
    }

    // Solve the eigenproblem to get the dominant eigenvalue and the
    // associated eigenvector of the iteration matrix.
    Scalar lambda = 0.0;
    Teuchos::RCP<Vector> eigenvector;
    {
        typedef typename VT::multivector_type MV;
        typedef typename MT::operator_type OP;

        Teuchos::ParameterList anasazi_params = 
            const_cast<Teuchos::ParameterList&>(plist).sublist("Anasazi",true);
        int verbosity = Anasazi::Errors + Anasazi::Warnings + 
                        Anasazi::FinalSummary + Anasazi::TimingDetails;
        anasazi_params.set( "Verbosity", verbosity );
        anasazi_params.set( "Which", "LM" );

        Teuchos::RCP<Vector> work_vec = VT::clone( *x );
        VT::randomize( *work_vec );

        Teuchos::RCP<Anasazi::BasicEigenproblem<Scalar,MV,OP> > eigen_problem =
            Teuchos::rcp( new Anasazi::BasicEigenproblem<Scalar,MV,OP>(
                              Teuchos::rcp_implicit_cast<OP>(reduced_H), 
                              Teuchos::rcp_implicit_cast<MV>(work_vec) ) );
        eigen_problem->setNEV( 1 );
        eigen_problem->setProblem();

        Anasazi::BlockKrylovSchurSolMgr<Scalar,MV,OP> solver_manager(
            eigen_problem, anasazi_params );
        solver_manager.solve();

        Anasazi::Eigensolution<Scalar,MV > sol = eigen_problem->getSolution();
        std::vector<Anasazi::Value<Scalar> > evals = sol.Evals;
        Teuchos::RCP<MV> evecs = sol.Evecs;

        MCLS_INSIST( sol.numVecs > 0, "No eigenvectors computed!" );

        lambda = std::pow( evals[0].realpart*evals[0].realpart +
                           evals[0].imagpart*evals[0].imagpart, 0.5 );

        eigenvector = VT::getVectorNonConst( *evecs, 0 );
    }

    // Generate the Monte Carlo domain.
    {
        // Generate the overlap for the transpose operator.
        Teuchos::RCP<Matrix> reduced_H_overlap;
        Teuchos::RCP<Vector> eigenvector_overlap;
        Teuchos::RCP<Vector> recovered_weights_overlap;
        if ( num_overlap > 0 )
        {
            reduced_H_overlap = 
                MT::copyNearestNeighbors( *reduced_H, num_overlap );

            eigenvector_overlap =
                MT::cloneVectorFromMatrixRows( *reduced_H_overlap );
            {
                VectorExport<Vector> evector_export( 
                    eigenvector, eigenvector_overlap );
                evector_export.doExportInsert();
            }

            recovered_weights_overlap =
                MT::cloneVectorFromMatrixRows( *reduced_H_overlap );
            {
                VectorExport<Vector> rweight_export( 
                    recovered_weights, recovered_weights_overlap );
                rweight_export.doExportInsert();
            }
        }

        // Get the total number of local rows.
        int num_rows = MT::getLocalNumRows( *reduced_H );
        if ( num_overlap > 0 )
        { 
            num_rows += MT::getLocalNumRows( *reduced_H_overlap );
        }

        // Allocate space in local row data arrays.
        d_columns = 
            Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > >( num_rows );
        d_cdfs = Teuchos::ArrayRCP<Teuchos::Array<double> >( num_rows );
        d_weights = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );
        d_h = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );

        // Build the local CDFs and weights.
        addMatrixToDomain( reduced_H, lambda, eigenvector, recovered_weights, 
                           tally_states, abs_probability );
        if ( num_overlap > 0 )
        {
            addMatrixToDomain( reduced_H_overlap, lambda, eigenvector_overlap,
                               recovered_weights_overlap, 
                               tally_states, abs_probability );
        }

        // Get the boundary states and their owning process ranks.
        if ( num_overlap == 0 )
        {
            buildBoundary( reduced_H, reduced_H );
        }
        else
        {
            buildBoundary( reduced_H_overlap, reduced_H );
        }
    }

    // By building the boundary data, now we know where we are sending
    // data. Find out who we are receiving from.
    Tpetra::Distributor distributor( MT::getComm(*A) );
    distributor.createFromSends( d_send_ranks() );
    d_receive_ranks = distributor.getImagesFrom();

    // Create the tally vector.
    Teuchos::Array<Ordinal> local_tally_states( tally_states.size() );
    std::copy( tally_states.begin(), tally_states.end(),
               local_tally_states.begin() );
    tally_states.clear();
    Teuchos::RCP<Vector> x_tally =
        VT::createFromRows( MT::getComm(*A), local_tally_states() );

    // Create the tally.
    d_tally = Teuchos::rcp( new TallyType(x, x_tally, d_estimator) );

    // If we are using the expected value estimator, provide the iteration
    // matrix to the tally. 
    if ( Estimator::EXPECTED_VALUE == d_estimator )
    {
        d_tally->setIterationMatrix( d_h, d_columns, d_row_indexer );
    }

    MCLS_ENSURE( !d_tally.is_null() );
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
template<class Vector, class Matrix>
MaximalEntropyDomain<Vector,Matrix>::MaximalEntropyDomain( 
    const Teuchos::ArrayView<char>& buffer,
    const Teuchos::RCP<const Comm>& set_comm )
    : d_row_indexer( Teuchos::rcp(new MapType()) )
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

    // Unpack the number of tally rows in the tally.
    ds >> num_tally;
    MCLS_CHECK( num_tally > 0 );

    // Unpack the local row indexer by key-value pairs.
    Ordinal global_row = 0;
    int local_row = 0;
    for ( Ordinal n = 0; n < num_rows; ++n )
    {
	ds >> global_row >> local_row;
	(*d_row_indexer)[global_row] = local_row;
    }

    // Unpack the local columns.
    std::set<Ordinal> im_unique_cols;
    d_columns = 
        Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > >( num_rows );
    Ordinal num_cols = 0;
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::iterator column_it;
    typename Teuchos::Array<Ordinal>::iterator index_it;
    for( column_it = d_columns.begin(); 
	 column_it != d_columns.end(); 
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
    d_weights = Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >( num_rows );
    num_values = 0;
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::iterator weight_it;
    Teuchos::ArrayRCP<double>::iterator weight_val_it;
    for( weight_it = d_weights.begin(); 
         weight_it != d_weights.end(); 
         ++weight_it )
    {
        // Unpack the number of entries in the row h.
        ds >> num_values;
        *weight_it = Teuchos::ArrayRCP<double>( num_values );

        // Unpack the iteration matrix values.
        for ( weight_val_it = weight_it->begin();
              weight_val_it != weight_it->end();
              ++weight_val_it )
        {
            ds >> *weight_val_it;
        }
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

    // Unpack the tally tally rows.
    Teuchos::Array<Ordinal> tally_rows( num_tally );
    typename Teuchos::Array<Ordinal>::iterator tally_it;
    for ( tally_it = tally_rows.begin();
	  tally_it != tally_rows.end();
	  ++tally_it )
    {
	ds >> *tally_it;
    }

    MCLS_CHECK( ds.end() == ds.getPtr() );

    // Create the tally.
    Teuchos::RCP<Vector> base_x = 
	VT::createFromRows( set_comm, base_rows() );
    Teuchos::RCP<Vector> tally_x = 
	VT::createFromRows( set_comm, tally_rows() );
    d_tally = Teuchos::rcp( new TallyType(base_x, tally_x, d_estimator) );

    // Set the iteration matrix data with the tally if using the expected
    // value estimator.
    if ( Estimator::EXPECTED_VALUE == d_estimator )
    {
        d_tally->setIterationMatrix( d_h, d_columns, d_row_indexer );
    }

    MCLS_ENSURE( !d_tally.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Pack the domain into a buffer.
 */
template<class Vector, class Matrix>
Teuchos::Array<char> MaximalEntropyDomain<Vector,Matrix>::pack() const
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
    s << Teuchos::as<Ordinal>(d_row_indexer->size());

    // Pack in the number of receive neighbors.
    s << Teuchos::as<int>(d_receive_ranks.size());

    // Pack in the number of send neighbors.
    s << Teuchos::as<int>(d_send_ranks.size());

    // Pack in the number of boundary states.
    s << Teuchos::as<Ordinal>(d_bnd_to_neighbor.size());

    // Pack in the number of base rows in the tally.
    s << Teuchos::as<Ordinal>(d_tally->numBaseRows());

    // Pack in the number of tally rows in the tally.
    s << Teuchos::as<Ordinal>(d_tally->numTallyRows());

    // Pack up the local row indexer by key-value pairs.
    typename MapType::const_iterator row_index_it;
    for ( row_index_it = d_row_indexer->begin();
	  row_index_it != d_row_indexer->end();
	  ++row_index_it )
    {
	s << row_index_it->first << row_index_it->second;
    }

    // Pack up the local columns.
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator column_it;
    typename Teuchos::Array<Ordinal>::const_iterator index_it;
    for( column_it = d_columns.begin(); 
	 column_it != d_columns.end(); 
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
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::const_iterator weight_it;
    Teuchos::ArrayRCP<double>::const_iterator weight_val_it;
    for( weight_it = d_weights.begin(); 
         weight_it != d_weights.end(); 
         ++weight_it )
    {
        // Pack the number of entries in the row.
        s << Teuchos::as<Ordinal>( weight_it->size() );

        // Pack the iteration matrix values.
        for ( weight_val_it = weight_it->begin();
              weight_val_it != weight_it->end();
              ++weight_val_it )
        {
            s << *weight_val_it;
        }
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
    typename MapType::const_iterator bnd_it;
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

    // Pack up the tally tally rows.
    Teuchos::Array<Ordinal> tally_rows = d_tally->tallyRows();
    typename Teuchos::Array<Ordinal>::const_iterator tally_it;
    for ( tally_it = tally_rows.begin();
	  tally_it != tally_rows.end();
	  ++tally_it )
    {
	s << *tally_it;
    }

    MCLS_ENSURE( s.end() == s.getPtr() );

    return buffer;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the size of this object in packed bytes.
 */
template<class Vector, class Matrix>
std::size_t MaximalEntropyDomain<Vector,Matrix>::getPackedBytes() const
{
    Serializer s;
    s.computeBufferSizeMode();

    // Pack the estimator type.
    s << Teuchos::as<int>(d_estimator);

    // Pack the local number of rows.
    s << Teuchos::as<Ordinal>(d_row_indexer->size());

    // Pack in the number of receive neighbors.
    s << Teuchos::as<int>(d_receive_ranks.size());

    // Pack in the number of send neighbors.
    s << Teuchos::as<int>(d_send_ranks.size());

    // Pack in the number of boundary states.
    s << Teuchos::as<Ordinal>(d_bnd_to_neighbor.size());

    // Pack in the number of base rows in the tally.
    s << Teuchos::as<Ordinal>(d_tally->numBaseRows());

    // Pack in the number of tally rows in the tally.
    s << Teuchos::as<Ordinal>(d_tally->numTallyRows());

    // Pack up the local row indexer by key-value pairs.
    typename MapType::const_iterator row_index_it;
    for ( row_index_it = d_row_indexer->begin();
	  row_index_it != d_row_indexer->end();
	  ++row_index_it )
    {
	s << row_index_it->first << row_index_it->second;
    }

    // Pack up the local columns.
    typename Teuchos::ArrayRCP<
        Teuchos::RCP<Teuchos::Array<Ordinal> > >::const_iterator column_it;
    typename Teuchos::Array<Ordinal>::const_iterator index_it;
    for( column_it = d_columns.begin(); 
	 column_it != d_columns.end(); 
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
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> >::const_iterator weight_it;
    Teuchos::ArrayRCP<double>::const_iterator weight_val_it;
    for( weight_it = d_weights.begin(); 
         weight_it != d_weights.end(); 
         ++weight_it )
    {
        // Pack the number of entries in the row.
        s << Teuchos::as<Ordinal>( weight_it->size() );

        // Pack the iteration matrix values.
        for ( weight_val_it = weight_it->begin();
              weight_val_it != weight_it->end();
              ++weight_val_it )
        {
            s << *weight_val_it;
        }
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
    typename MapType::const_iterator bnd_it;
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

    // Pack up the tally tally rows.
    Teuchos::Array<Ordinal> tally_rows = d_tally->tallyRows();
    typename Teuchos::Array<Ordinal>::const_iterator tally_it;
    for ( tally_it = tally_rows.begin();
	  tally_it != tally_rows.end();
	  ++tally_it )
    {
	s << *tally_it;
    }

    return s.size();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank from which we will receive.
 */
template<class Vector, class Matrix>
int MaximalEntropyDomain<Vector,Matrix>::receiveNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < d_receive_ranks.size() );
    return d_receive_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank to which we will send.
 */
template<class Vector, class Matrix>
int MaximalEntropyDomain<Vector,Matrix>::sendNeighborRank( int n ) const
{
    MCLS_REQUIRE( n >= 0 && n < d_send_ranks.size() );
    return d_send_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain that owns a boundary state (local neighbor
 * id).
 */
template<class Vector, class Matrix>
int MaximalEntropyDomain<Vector,Matrix>::owningNeighbor( const Ordinal& state ) const
{
    typename MapType::const_iterator neighbor = d_bnd_to_neighbor.find( state );
    MCLS_REQUIRE( neighbor != d_bnd_to_neighbor.end() );
    return neighbor->second;
}

//---------------------------------------------------------------------------//
/*
 * \brief Add matrix data to the local domain.
 */
template<class Vector, class Matrix>
void MaximalEntropyDomain<Vector,Matrix>::addMatrixToDomain( 
    const Teuchos::RCP<const Matrix>& A,
    const Scalar lambda,
    const Teuchos::RCP<const Vector>& eigenvector,
    const Teuchos::RCP<const Vector>& recovered_weights,
    std::set<Ordinal>& tally_states,
    const double abs_probability )
{
    MCLS_REQUIRE( !A.is_null() );

    Teuchos::ArrayRCP<const double> eigenvector_view =
        VT::view( *eigenvector );
    Ordinal local_num_rows = MT::getLocalNumRows( *A );
    Ordinal global_row = 0;
    int offset = d_row_indexer->size();
    int max_entries = MT::getGlobalMaxNumRowEntries( *A );
    std::size_t num_entries = 0;
    Teuchos::Array<double>::iterator cdf_iterator;
    double pdf_norm = 1.0 - abs_probability;
    Ordinal local_j = 0;

    for ( Ordinal i = 0; i < local_num_rows; ++i )
    {
	// Add the global row id and local row id to the indexer.
	global_row = MT::getGlobalRow(*A, i);
	(*d_row_indexer)[global_row] = i+offset;

	// Allocate column and CDF memory for this row.
        d_columns[i+offset] = 
            Teuchos::rcp( new Teuchos::Array<Ordinal>(max_entries) );
	d_cdfs[i+offset].resize( max_entries );

	// Add the columns and base PDF values for this row.
	MT::getGlobalRowCopy( *A, 
			      global_row,
			      (*d_columns[i+offset])(), 
			      d_cdfs[i+offset](),
			      num_entries );

	// Check for degeneracy.
	MCLS_CHECK( num_entries > 0 );

	// Resize local column and CDF arrays for this row.
	d_columns[i+offset]->resize( num_entries );
	d_cdfs[i+offset].resize( num_entries );

        // Save the current cdf state as the iteration matrix.
        d_h[i+offset] = Teuchos::ArrayRCP<double>( d_cdfs[i+offset].size() );
        std::copy( d_cdfs[i+offset].begin(), d_cdfs[i+offset].end(),
                   d_h[i+offset].begin() );

        // Allocate for the weight matrix row.
        d_weights[i+offset] = 
            Teuchos::ArrayRCP<double>( d_cdfs[i+offset].size() );

        // Build the probabilities and weights for the maximal entropy random
        // walk.
        for ( std::size_t j = 0; j < num_entries; ++j )
        {
            local_j = VT::getLocalRow( *eigenvector, (*d_columns[i+offset])[j] );
            d_weights[i+offset][j] = lambda * eigenvector_view[i] / 
                                     ( eigenvector_view[local_j] * pdf_norm );
            d_cdfs[i+offset][j] /= d_weights[i+offset][j];
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

        // Depending on how tightly the eigenvalue problem was converged, we
        // may get a value here slightly larger than 1 for the CDF. Just set
        // it to 1 here. We'll check to make sure that we're at least close to
        // 1.
        if ( d_cdfs[i+offset].back() > 1.0 )
        {
            MCLS_CHECK( std::abs(1.0 - d_cdfs[i+offset].back()) <= 1.0e-4 );
            d_cdfs[i+offset].back() = 1.0;
        }

        // Add the absorbing state.
        d_cdfs[i+offset].push_back( 1.0 );
        d_columns[i+offset]->push_back( 
            Teuchos::OrdinalTraits<Ordinal>::invalid() );

        // If we're using the collision estimator, add the global row as a
        // local tally state.
        if ( Estimator::COLLISION == d_estimator )
        {
            tally_states.insert( global_row );
        }
        // Else if we're using the expected value estimator add the columns from
        // this row as local tally states. Do not insert the absorbing state.
        else if ( Estimator::EXPECTED_VALUE == d_estimator )
        {
            typename Teuchos::Array<Ordinal>::const_iterator col_it;
            for ( col_it = d_columns[i+offset]->begin();
                  col_it != d_columns[i+offset]->end()-1;
                  ++col_it )
            {
                tally_states.insert( *col_it );
            }
        }
        else
        {
            MCLS_INSIST( Estimator::COLLISION == d_estimator || 
                         Estimator::EXPECTED_VALUE == d_estimator,
                         "Unsupported estimator type" );
        }
    }
}

//---------------------------------------------------------------------------//
/*
 * \brief Build boundary data.
 */
template<class Vector, class Matrix>
void MaximalEntropyDomain<Vector,Matrix>::buildBoundary( 
    const Teuchos::RCP<const Matrix>& A,
    const Teuchos::RCP<const Matrix>& base_A )
{
    MCLS_REQUIRE( !A.is_null() );

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

#endif // end MCLS_MAXIMALENTROPYDOMAIN_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_MaximalEntropyDomain_impl.hpp
// ---------------------------------------------------------------------------//
