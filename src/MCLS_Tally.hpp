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
 * \file MCLS_Tally.hpp
 * \author Stuart R. Slattery
 * \brief Tally declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTDOMAIN_HPP
#define MCLS_ADJOINTDOMAIN_HPP

#include <MCLS_DBC.hpp>
#include <MCLS_History.hpp>
#include <MCLS_VectorExport.hpp>
#include <MCLS_VectorTraits.hpp>

#include <Teuchos_RCP.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class Tally
 * \brief Monte Carlo tally for the linear system solution vector.
 */
template<class Vector>
class Tally
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                               vector_type;
    typedef VectorTraits<Vector>                                 VT;
    typedef History<VT::scalar_type,VT::global_ordinal_type>     HistoryType;
    //@}

    // Constructor.
    Tally( const Teuchos::RCP<Vector>& x, 
	   const Teuchos::RCP<Vector>& x_overlap );

    // Destructor.
    ~Tally()
    { /* ... */ }

    // Add a history's contribution to the tally.
    void tallyHistory( const HistoryType& history );

    // Combine the overlap tally with the base decomposition tally.
    void 

  private:

    // Solution vector in original decomposition.
    Teuchos::RCP<Vector>& d_x;

    // Solution vector in overlap decomposition.
    Teuchos::RCP<Vector>& d_x_overlap;

    // Overlap to original decomposition vector export.
    VectorExport<Vector> d_export;
};

//---------------------------------------------------------------------------//

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Process a history through a transition to a new state.
 */
template<class Scalar, class Ordinal>
inline void Tally<Scalar,Ordinal>::processTransition( 
    history_type& history )
{
    Require( isLocalState( history.state() ) );

    history.setState( 
	d_columns[d_row_indexer.get(history.state())][ 
	    SamplingTools::sampleDiscreteCDF( 
		d_cdfs[d_row_indexer.get(history.state()) ](),
		history.rng.random() )] );

    history.multiplyWeight( d_weights[d_row_indexer.get(history.state())] );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given state is on-process.
 */
template<class Scalar, class Ordinal>
inline bool 
Tally<Scalar,Ordinal>::isLocalState( const Ordinal& state )
{
    return d_row_indexer.containsKey( state );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain process rank.
 */
template<class Scalar, class Ordinal>
inline int Tally<Scalar,Ordinal>::neighborRank( int n ) const
{
    Require( n >= 0 && n < d_neighbor_ranks.size() );
    return d_neighbor_ranks[n];
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the neighbor domain that owns a boundary state (local neighbor
 * id).
 */
template<class Scalar, class Ordinal>
inline int Tally<Scalar,Ordinal>::owningNeighbor( const Ordinal& state )
{
    Require( d_bnd_to_neighbor.containsKey(state) );
    return d_bnd_to_neighbor.get(state);
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_Tally_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTDOMAIN_HPP

//---------------------------------------------------------------------------//
// end MCLS_Tally.hpp
// ---------------------------------------------------------------------------//

