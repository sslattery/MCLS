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
 * \file MCLS_AlmostOptimalDomain.hpp
 * \author Stuart R. Slattery
 * \brief AlmostOptimalDomain declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ALMOSTOPTIMALDOMAIN_HPP
#define MCLS_ALMOSTOPTIMALDOMAIN_HPP

#include <stack>
#include <set>
#include <unordered_map>
#include <random>

#include "MCLS_DBC.hpp"
#include "MCLS_SamplingTools.hpp"
#include "MCLS_Events.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_PRNG.hpp"
#include "MCLS_RNGTraits.hpp"
#include "MCLS_HistoryTraits.hpp"
#include "MCLS_Serializer.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_OrdinalTraits.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class AlmostOptimalDomain
 * \brief Almost optimal Monte Carlo transport domain.
 *
 * Derived from the Neumann-Ulam decomposition of a matrix:
 *
 * H = I - A
 * H = (P) x (W)
 *
 * where the transition probabilities are relative to the iteration matrix
 * components.
 *
 * This domain contains data for all local states in the system, including the
 * overlap and neighboring domains. This object is responsible for creating
 * the tally for the solution vector over the domain as it has ownership of
 * the parallel decomposition of the domain.
 *
 * For all estimator types, the AlmostOptimalDomain constructs weights for all
 * transitions, cumulative distribution functions for each local state in the
 * system that can be sampled, and the associated states for those CDFs to
 * which a given initial state in the local system can transition to.
 */
template<class Vector, class Matrix, class RNG, class Tally>
class AlmostOptimalDomain
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                        vector_type;
    typedef VectorTraits<Vector>                          VT;
    typedef Matrix                                        matrix_type;
    typedef MatrixTraits<Vector,Matrix>                   MT;
    typedef typename VT::global_ordinal_type              Ordinal;
    typedef Tally                                         tally_type;
    typedef typename tally_type::HistoryType              HistoryType;
    typedef HistoryTraits<HistoryType>                    HT;
    typedef std::stack<Teuchos::RCP<HistoryType> >        BankType;
    typedef Teuchos::Comm<int>                            Comm;
    typedef RNG                                           rng_type;
    typedef RNGTraits<RNG>                                RNGT;
    typedef typename RNGT::uniform_real_distribution_type RandomDistribution;
    typedef RandomDistributionTraits<RandomDistribution>  RDT;
    //@}

    // Default constructor.
    AlmostOptimalDomain();

    // Destructor.
    virtual ~AlmostOptimalDomain()
    { /* ... */ }

    // Build the domain.
    void buildDomain( const Teuchos::RCP<const Matrix>& A,
		      const Teuchos::RCP<Vector>& x,
		      const Teuchos::ParameterList& plist,
		      Teuchos::Array<Ordinal>& local_tally_states );

    // Set the random number generator.
    void setRNG( const Teuchos::RCP<PRNG<RNG> >& rng )
    { b_rng = rng; }

    // Pack the domain into a buffer.
    void packDomain( Serializer& s ) const;

    // Unpack the domain from a buffer.
    void unpackDomain( Deserializer& ds, Teuchos::Array<Ordinal>& base_rows );

    //! Set the weight cutoff.
    void setCutoff( const double weight_cutoff )
    { b_weight_cutoff = weight_cutoff; }

    // Given a history with a global state in the local domain, set the local
    // state of that history.
    inline void setHistoryLocalState( HistoryType& history ) const;

    // Process a history through a transition to a new state.
    inline void processTransition( HistoryType& history ) const;

    //! Deterimine if a history should be terminated.
    inline bool terminateHistory( const HistoryType& history ) const
    { return HT::weightAbs(history) < b_weight_cutoff; }

    // Get the domain tally.
    Teuchos::RCP<Tally> domainTally() const
    { return b_tally; }

    // Determine if a given state global is in the local domain.
    inline bool isGlobalState( const Ordinal& state ) const;

    // Determine if a given state is on the boundary.
    inline bool isBoundaryState( const Ordinal& state ) const;

    //! Get the number of neighboring domains from which we will receive.
    int numReceiveNeighbors() const
    { return b_receive_ranks.size(); }

    // Get the neighbor domain process rank from which we will receive.
    int receiveNeighborRank( int n ) const;

    //! Get the number of neighboring domains to which we will send.
    int numSendNeighbors() const
    { return b_send_ranks.size(); }

    // Get the neighbor domain process rank to which we will send.
    int sendNeighborRank( int n ) const;

    // Get the neighbor domain that owns a boundary state (local neighbor id).
    int owningNeighbor( const Ordinal& state ) const;

    // Get the local states in the domain.
    Teuchos::Array<Ordinal> localStates() const;

  private:

    // Add matrix data to the local domain.
    void addMatrixToDomain( const Teuchos::RCP<const Matrix>& A,
                            std::set<Ordinal>& tally_states,
			    const double relaxation );

    // Build boundary data.
    void buildBoundary( const Teuchos::RCP<const Matrix>& A,
			const Teuchos::RCP<const Matrix>& base_A );

  protected:

    // Random number generator.
    Teuchos::RCP<PRNG<RNG> > b_rng;

    // Random number distribution.
    Teuchos::RCP<RandomDistribution> b_rng_dist;

    // Monte Carlo estimator type.
    int b_estimator;

    // History weight cutoff.
    double b_weight_cutoff;

    // Domain tally.
    Teuchos::RCP<Tally> b_tally;

    // Global-to-local row indexer.
    std::unordered_map<Ordinal,int> b_g2l_row_indexer;

    // Local CDF columns in global indexing.
    Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > > b_global_columns;

    // Local CDF columns in local indexing.
    Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<int> > > b_local_columns;

    // Local CDF values.
    Teuchos::ArrayRCP<Teuchos::Array<double> > b_cdfs;

    // Local iteration matrix values.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > b_h;

    // Local weights.
    Teuchos::ArrayRCP<double> b_weights;

    // Neighboring domain process ranks from which we will receive.
    Teuchos::Array<int> b_receive_ranks;

    // Neighboring domain process ranks to which we will send.
    Teuchos::Array<int> b_send_ranks;

    // Boundary state to owning neighbor local id table.
    std::unordered_map<Ordinal,int> b_bnd_to_neighbor;
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Given a history with a global state in the local domain, set the
 * local state of that history.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::setHistoryLocalState( 
    HistoryType& history ) const
{
    MCLS_REQUIRE( isGlobalState(HT::globalState(history)) );
    MCLS_REQUIRE( b_g2l_row_indexer.count(HT::globalState(history)) );
    HT::setLocalState( 
	history, b_g2l_row_indexer.find(HT::globalState(history))->second );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Process a history through a transition to a new state.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::processTransition( 
    HistoryType& history ) const
{
    MCLS_REQUIRE( Teuchos::nonnull(b_rng) );
    MCLS_REQUIRE( HT::alive(history) );
    MCLS_REQUIRE( Event::TRANSITION == HT::event(history) );
    MCLS_REQUIRE( isGlobalState(HT::globalState(history)) );

    // Get the incoming state.
    int in_state = history.localState();

    // Sample the row CDF to get a new outgoing state.
    int out_state = 
	SamplingTools::sampleDiscreteCDF( b_cdfs[in_state].getRawPtr(),
					  b_cdfs[in_state].size(),
					  b_rng->random(*b_rng_dist) );

    // Set the new local state with the history.
    HT::setLocalState( history, (*b_local_columns[in_state])[out_state] );

    // Set the new global state with the history.
    HT::setGlobalState( history, (*b_global_columns[in_state])[out_state] );

    // Update the history weight with the transition weight.
    int transition_sign = (b_h[in_state][out_state] > 0.0) - 
			  (b_h[in_state][out_state] < 0.0);
    HT::multiplyWeight( history, b_weights[in_state]*transition_sign );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given global state is in the local domain.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline bool AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::isGlobalState( 
    const Ordinal& state ) const
{
    return b_g2l_row_indexer.count( state );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given global state is on the boundary.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline bool AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::isBoundaryState( 
    const Ordinal& state ) const
{
    return b_bnd_to_neighbor.count( state );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AlmostOptimalDomain_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ALMOSTOPTIMALDOMAIN_HPP

//---------------------------------------------------------------------------//
// end MCLS_AlmostOptimalDomain.hpp
// ---------------------------------------------------------------------------//

