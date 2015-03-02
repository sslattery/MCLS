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

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector>
ForwardTally<Vector>::ForwardTally( const Teuchos::RCP<Vector>& x )
    : d_x( x )
{ 
    MCLS_ENSURE( Teuchos::nonnull(d_x) );
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
    d_b_view = VT::view( *d_b );
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

    // If the history starting state has already been tallied, add the history
    // tally sum to the local sum and increment the tally count for the
    // starting state.
    if ( d_states_values_counts.count(history.startingState()) )
    {
	auto state_val_count =
	    d_states_values_counts.find( history.startingState() );
	state_val_count->second.first += history.historyTally();
	state_val_count->second.second += 1;
    }

    // Otherwise add the history state to the local states, sum, and count.
    else
    {
	d_states_values_counts.emplace(
	    history.startingState(), std::make_pair(history.historyTally(),1) );
    }
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
 * \brief Zero out tally data.
 */
template<class Vector>
void ForwardTally<Vector>::zeroOut()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_x) );
    VT::putScalar( *d_x, 0.0 );
    d_states_values_counts.clear();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Combine the overlap tally with the operator decomposition tally in
 * the set and normalize by the counted number of histories in each state.
 */
template<class Vector>
void ForwardTally<Vector>::finalize()
{
    // Extract the tally states.
    Teuchos::Array<Ordinal> tally_states( d_states_values_counts.size() );
    typename Teuchos::Array<Ordinal>::iterator state_it;
    typename std::unordered_map<Ordinal,std::pair<Scalar,int> >::const_iterator
	svc_it;
    for ( state_it = tally_states.begin(),
	    svc_it = d_states_values_counts.begin();
	  state_it != tally_states.end();
	  ++state_it, ++svc_it )
    {
	*state_it = svc_it->first;
    }
    
    // Build a vector from the dead history states.
    Teuchos::RCP<Vector> x_tally = 
	VT::createFromRows( VT::getComm(*d_x), tally_states() );

    // Copy the tally data into the vector.
    for ( auto svc : d_states_values_counts )
    {
	VT::sumIntoGlobalValue( *x_tally, svc.first, svc.second.first );
    }

    // Export the local history vector to the base vector.
    {
        VectorExport<Vector> tally_exporter( x_tally, d_x );
        tally_exporter.doExportAdd();
    }

    // Build a vector from the dead history states.
    Teuchos::RCP<Vector> count_tally = 
	VT::createFromRows( VT::getComm(*d_x), tally_states() );
 
    // Copy the tally counts into the vector.
    for ( auto svc : d_states_values_counts )
    {
	VT::sumIntoGlobalValue( *count_tally, svc.first, svc.second.second );
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

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_FORWARDTALLY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_ForwardTally_impl.hpp
// ---------------------------------------------------------------------------//

