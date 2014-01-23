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
 * \file MCLS_ForwardTally_impl.hpp
 * \author Stuart R. Slattery
 * \brief ForwardTally implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FORWARDTALLY_IMPL_HPP
#define MCLS_FORWARDTALLY_IMPL_HPP

#include <algorithm>

#include "MCLS_VectorExport.hpp"
#include "MCLS_Events.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_CommHelpers.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector>
ForwardTally<Vector>::ForwardTally( const Teuchos::RCP<Vector>& x,
                                    const int estimator )
    : d_x( x )
    , d_estimator( estimator )
{ 
    MCLS_ENSURE( Teuchos::nonnull(d_x) );
    MCLS_ENSURE( Estimator::COLLISION == estimator );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Assign the source vector to the tally. This vector should contain
 * both the base and overlap rows.
 */
template<class Vector>
void ForwardTally<Vector>::setSource( const Teuchos::RCP<Vector>& b )
{
    MCLS_REQUIRE( Teuchos::nonnull(b) );
    d_b = b;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Post-process a history if it is permanently killed in the local
 * domain. 
 */
template<class Vector>
void ForwardTally<Vector>::postProcessHistory( const HistoryType& history )
{
    MCLS_REQUIRE( !history.alive() );
    MCLS_REQUIRE( Event::CUTOFF == history.event() );
    MCLS_REQUIRE( d_tally_states.size() == d_tally_values.size() );

    // If the history starting state has already been tallied, add the history
    // tally sum to the local sum and increment the tally count for the
    // starting state.
    typename Teuchos::Array<Ordinal>::iterator state_it = 
	std::find( d_tally_states.begin(), d_tally_states.end(), 
		   history.startingState() );
    if ( state_it != d_tally_states.end() )
    {
        typename VT::local_ordinal_type local_tally_state = 
            std::distance(d_tally_states.begin(),state_it);
	d_tally_values[ local_tally_state ] += history.historyTally();
        d_tally_count[ local_tally_state ] += 1;
    }

    // Otherwise add the history state to the local states, sum, and count.
    else
    {
	d_tally_states.push_back( history.startingState() );
	d_tally_values.push_back( history.historyTally() );
        d_tally_count.push_back( 1 );
    }

    MCLS_ENSURE( d_tally_states.size() == d_tally_values.size() );
    MCLS_ENSURE( d_tally_count.size() == d_tally_values.size() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the overlap tally with the operator decomposition tally in
 * the set and normalize by the counted number of histories in each state.
 */
template<class Vector>
void ForwardTally<Vector>::combineSetTallies( 
    const Teuchos::RCP<const Comm>& set_comm )
{
    // Build a vector from the dead history states.
    Teuchos::RCP<Vector> x_tally = 
	VT::createFromRows( set_comm, d_tally_states() );

    // Copy the tally data into the vector.
    typename Teuchos::Array<Scalar>::const_iterator val_it;
    typename Teuchos::Array<Ordinal>::const_iterator state_it;
    for ( state_it = d_tally_states.begin(),
	    val_it = d_tally_values.begin();
	  state_it != d_tally_states.end();
	  ++state_it, ++val_it )
    {
	VT::sumIntoGlobalValue( *x_tally, *state_it, *val_it );
    }

    // Export the local history vector to the base vector.
    {
        VectorExport<Vector> tally_exporter( x_tally, d_x );
        tally_exporter.doExportAdd();
    }

    // Build a vector from the dead history states.
    Teuchos::RCP<Vector> count_tally = 
	VT::createFromRows( set_comm, d_tally_states() );
 
    // Copy the tally counts into the vector.
    typename Teuchos::Array<int>::const_iterator count_it;
    for ( state_it = d_tally_states.begin(),
	    count_it = d_tally_count.begin();
	  state_it != d_tally_states.end();
	  ++state_it, ++count_it )
    {
	VT::sumIntoGlobalValue( *count_tally, *state_it, 
                                Teuchos::as<Scalar>(*count_it) );
    }

    // Build a vector from the base states.
    Teuchos::RCP<Vector> count_base = VT::clone( *d_x );

    // Export add the local history vector to the base vector.
    {
        VectorExport<Vector> count_exporter( count_tally, count_base );
        count_exporter.doExportAdd();
    }
   
    // Normalize each state in the local tally vector by the count.
    Teuchos::ArrayRCP<const Scalar> count_view = VT::view( *count_base );
    Teuchos::ArrayRCP<Scalar> x_view = VT::viewNonConst( *d_x );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator norm_it;
    typename Teuchos::ArrayRCP<Scalar>::iterator x_it;
    for ( x_it = x_view.begin(), norm_it = count_view.begin();
          x_it != x_view.end();
          ++x_it, ++norm_it )
    {
        MCLS_CHECK( *norm_it >= 0.0 );

        if ( *norm_it > 0.0 )
        {
            *x_it /= *norm_it;
        }
        else
        {
            MCLS_CHECK( 0.0 == *x_it );
            *x_it = 0.0;
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the base tallies across a block and normalize by the number
 * of sets.
 */
template<class Vector>
void ForwardTally<Vector>::combineBlockTallies(
    const Teuchos::RCP<const Comm>& block_comm, const int num_sets )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_x) );
    MCLS_REQUIRE( !block_comm.is_null() );

    Teuchos::ArrayRCP<const Scalar> const_tally_view = VT::view( *d_x );

    Teuchos::ArrayRCP<Scalar> copy_buffer( const_tally_view.size() );

    Teuchos::reduceAll<int,Scalar>( *block_comm,
				    Teuchos::REDUCE_SUM,
				    Teuchos::as<int>( const_tally_view.size() ),
				    const_tally_view.getRawPtr(),
				    copy_buffer.getRawPtr() );

    Teuchos::ArrayRCP<Scalar> tally_view = VT::viewNonConst( *d_x );
    
    std::copy( copy_buffer.begin(), copy_buffer.end(), tally_view.begin() );
    VT::scale( *d_x, 1.0 / Teuchos::as<double>(num_sets) );
}

//---------------------------------------------------------------------------//
/*
 * \brief Normalize base decomposition tally with the number of specified
 * histories.
 */
template<class Vector>
void ForwardTally<Vector>::normalize( const int& nh )
{
    VT::scale( *d_x, 1.0 );
}

//---------------------------------------------------------------------------//
/*
 * \brief Set the base tally vector.
 */
template<class Vector>
void ForwardTally<Vector>::setBaseVector( const Teuchos::RCP<Vector>& x_base )
{
    MCLS_REQUIRE( Teuchos::nonnull(x_base) );
    d_x = x_base;
}

//---------------------------------------------------------------------------//
/*
 * \brief Zero out tally data.
 */
template<class Vector>
void ForwardTally<Vector>::zeroOut()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_x) );
    VT::putScalar( *d_x, Teuchos::ScalarTraits<Scalar>::zero() );
    d_tally_states.clear();
    d_tally_values.clear();
    d_tally_count.clear();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the number global rows in the base decomposition.
 */
template<class Vector>
typename ForwardTally<Vector>::Ordinal 
ForwardTally<Vector>::numBaseRows() const
{
    MCLS_CHECK( Teuchos::nonnull(d_x) );
    return VT::getLocalLength( *d_x );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the global rows in the base decomposition.
 */
template<class Vector>
Teuchos::Array<typename ForwardTally<Vector>::Ordinal>
ForwardTally<Vector>::baseRows() const
{
    MCLS_CHECK( Teuchos::nonnull(d_x) );

    Teuchos::Array<Ordinal> base_rows( VT::getLocalLength(*d_x) );
    typename Teuchos::Array<Ordinal>::iterator row_it;
    typename VT::local_ordinal_type local_row = 
	Teuchos::OrdinalTraits<typename VT::local_ordinal_type>::zero();
    for ( row_it = base_rows.begin();
	  row_it != base_rows.end();
	  ++row_it )
    {
	*row_it = VT::getGlobalRow( *d_x, local_row );
	++local_row;
    }

    return base_rows;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_FORWARDTALLY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_ForwardTally_impl.hpp
// ---------------------------------------------------------------------------//

