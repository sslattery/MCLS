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
#include "MCLS_DomainTraits.hpp"
#include "MCLS_SamplingTools.hpp"
#include "MCLS_Events.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_TallyTraits.hpp"
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

#include <Tpetra_CrsMatrix.hpp>

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
 * This domain contains data for all local states in the system. This object
 * is responsible for creating the tally for the solution vector over the
 * domain as it has ownership of the parallel decomposition of the domain.
 *
 * The AlmostOptimalDomain constructs weights for all transitions, cumulative
 * distribution functions for each local state in the system that can be
 * sampled, and the associated states for those CDFs to which a given initial
 * state in the local system can transition to.
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
    typedef Tally                                         TallyType;
    typedef TallyTraits<Tally>                            TT;
    typedef typename TT::history_type                     HistoryType;
    typedef HistoryTraits<HistoryType>                    HT;
    typedef std::stack<HistoryType>                       BankType;
    typedef RNG                                           rng_type;
    typedef RNGTraits<RNG>                                RNGT;
    typedef typename RNGT::uniform_real_distribution_type RandomDistribution;
    typedef RandomDistributionTraits<RandomDistribution>  RDT;
    //@}

    // Constructor.
    AlmostOptimalDomain( const Teuchos::RCP<const Matrix>& A,
			 const Teuchos::RCP<Vector>& x,
			 const Teuchos::ParameterList& plist );

    // Set the random number generator.
    void setRNG( const Teuchos::RCP<PRNG<RNG> >& rng )
    { d_rng = rng; }

    // Given a history with a global state in the local domain, set the local
    // state of that history.
    inline void setHistoryLocalState( HistoryType& history ) const;

    // Process a history through a transition to a new state.
    inline void processTransition( HistoryType& history ) const;

    // Determine if we should terminate the history.
    inline bool terminateHistory( const HistoryType& history ) const
    { return (HT::numSteps(history) >= d_history_length); }

    // Get the domain tally.
    Teuchos::RCP<Tally> domainTally() const
    { return d_tally; }

    // Determine if a given state global is in the local domain.
    inline bool isGlobalState( const Ordinal& state ) const;

    // Determine if a given state is on the boundary.
    inline bool isBoundaryState( const Ordinal& state ) const;

    //! Get the number of neighboring domains from which we will receive.
    int numReceiveNeighbors() const
    { return d_receive_ranks.size(); }

    // Get the neighbor domain process rank from which we will receive.
    int receiveNeighborRank( int n ) const;

    //! Get the number of neighboring domains to which we will send.
    int numSendNeighbors() const
    { return d_send_ranks.size(); }

    // Get the neighbor domain process rank to which we will send.
    int sendNeighborRank( int n ) const;

    // Get the neighbor domain that owns a boundary state (local neighbor id).
    int owningNeighbor( const Ordinal& state ) const;

    // Get the local states in the domain.
    Teuchos::Array<Ordinal> localStates() const;

    // Compute the spectral radius of H and H*.
    Teuchos::Array<double> computeConvergenceCriteria() const;

  private:

    // Build the domain.
    void buildDomain( const Teuchos::RCP<const Matrix>& A,
		      const Teuchos::ParameterList& plist );

    // Add matrix data to the local domain.
    void addMatrixToDomain( const Teuchos::RCP<const Matrix>& A,
			    const double relaxation );

    // Build boundary data.
    void buildBoundary( const Teuchos::RCP<const Matrix>& A );

    // Given a crs matrix, compute its spectral radius.
    double computeSpectralRadius( 
	const Teuchos::RCP<Tpetra::CrsMatrix<double,int,Ordinal> >& matrix ) const;

  protected:

    // Random number generator.
    Teuchos::RCP<PRNG<RNG> > d_rng;

    // Random number distribution.
    Teuchos::RCP<RandomDistribution> d_rng_dist;

    // History length.
    int d_history_length;

    // Domain tally.
    Teuchos::RCP<Tally> d_tally;

    // Global-to-local row indexer.
    std::unordered_map<Ordinal,int> d_g2l_row_indexer;

    // Local CDF columns in global indexing.
    Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > > d_global_columns;

    // Local CDF columns in local indexing.
    Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<int> > > d_local_columns;

    // Local CDF values.
    Teuchos::ArrayRCP<Teuchos::Array<double> > d_cdfs;

    // Local iteration matrix values.
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > d_h;

    // Local weights.
    Teuchos::ArrayRCP<double> d_weights;

    // Neighboring domain process ranks from which we will receive.
    Teuchos::Array<int> d_receive_ranks;

    // Neighboring domain process ranks to which we will send.
    Teuchos::Array<int> d_send_ranks;

    // Boundary state to owning neighbor local id table.
    std::unordered_map<Ordinal,int> d_bnd_to_neighbor;

    // Parallel communicator.
    Teuchos::RCP<const Teuchos::Comm<int> > d_comm;
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
    MCLS_REQUIRE( d_g2l_row_indexer.count(HT::globalState(history)) );
    HT::setLocalState( 
	history, d_g2l_row_indexer.find(HT::globalState(history))->second );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Process a history through a transition to a new state.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline void AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::processTransition( 
    HistoryType& history ) const
{
    MCLS_REQUIRE( Teuchos::nonnull(d_rng) );
    MCLS_REQUIRE( HT::alive(history) );
    MCLS_REQUIRE( Event::TRANSITION == HT::event(history) );
    MCLS_REQUIRE( isGlobalState(HT::globalState(history)) );

    // Get the incoming state.
    int in_state = history.localState();

    // Sample the row CDF to get a new outgoing state.
    int out_state = 
	SamplingTools::sampleDiscreteCDF( d_cdfs[in_state].getRawPtr(),
					  d_cdfs[in_state].size(),
					  d_rng->random(*d_rng_dist) );

    // Set the new local state with the history.
    HT::setLocalState( history, (*d_local_columns[in_state])[out_state] );

    // Set the new global state with the history.
    HT::setGlobalState( history, (*d_global_columns[in_state])[out_state] );

    // Update the history weight with the transition weight.
    int transition_sign = ( d_h[in_state][out_state] > 0.0 ) ? 1 : -1;
    HT::multiplyWeight( history, d_weights[in_state]*transition_sign );

    // Increment the history step count.
    HT::addStep( history );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given global state is in the local domain.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline bool AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::isGlobalState( 
    const Ordinal& state ) const
{
    return d_g2l_row_indexer.count( state );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given global state is on the boundary.
 */
template<class Vector, class Matrix, class RNG, class Tally>
inline bool AlmostOptimalDomain<Vector,Matrix,RNG,Tally>::isBoundaryState( 
    const Ordinal& state ) const
{
    return d_bnd_to_neighbor.count( state );
}

//---------------------------------------------------------------------------//
// DomainTraits implementation.
//---------------------------------------------------------------------------//
/*!
 * \class DomainTraits
 * \brief Traits implementation for the AlmostOptimalDomain.
 */
template<class Vector, class Matrix, class RNG, class Tally>
class DomainTraits<AlmostOptimalDomain<Vector,Matrix,RNG,Tally> >
{
  public:

    //@{
    //! Typedefs.
    typedef AlmostOptimalDomain<Vector,Matrix,RNG,Tally> domain_type;
    typedef typename domain_type::Ordinal                ordinal_type;
    typedef typename domain_type::HistoryType            history_type;
    typedef typename domain_type::TallyType              tally_type;
    typedef typename domain_type::BankType               bank_type;
    typedef typename domain_type::rng_type               rng_type;
    //@}

    /*!
     * \brief Set a random number generator with the domain.
     */
    static void setRNG( domain_type& domain,
			const Teuchos::RCP<PRNG<rng_type> >& rng )
    {
	domain.setRNG( rng );
    }

    /*!
     * \brief Given a history with a global state in the local domain, set the
     * local state of that history.
     */
    static inline void setHistoryLocalState( 
	const domain_type& domain, history_type& history )
    { 
	domain.setHistoryLocalState( history );
    }

    /*!
     * \brief Process a history through a transition in the local domain to a
     * new state
     */
    static inline void processTransition( 
	const domain_type& domain, history_type& history )
    { 
	domain.processTransition( history );
    }

    /*!
     * \brief Deterimine if a history should be terminated.
     */
    static inline bool terminateHistory( 
	const domain_type& domain, const history_type& history )
    { 
	return domain.terminateHistory( history );
    }

    /*!
     * \brief Get the tally associated with this domain.
     */
    static Teuchos::RCP<tally_type> domainTally( const domain_type& domain )
    { 
	return domain.domainTally();
    }

    /*!
     * \brief Determine if a given state is in the local domain.
     */
    static inline bool isGlobalState( const domain_type& domain, 
			      const ordinal_type state )
    { 
	return domain.isGlobalState( state );
    }

    /*!
     * \brief Determine if a given state is on the boundary.
     */
    static inline bool isBoundaryState( const domain_type& domain, 
                                 const ordinal_type state )
    { 
	return domain.isBoundaryState( state );
    }

    /*!
     * \brief Get the number of neighbors from which this domain will
     * receive. 
     */
    static int numReceiveNeighbors( const domain_type& domain )
    {
	return domain.numReceiveNeighbors();
    }

    /*!
     * \brief Given a local neighbor ID, return the proc rank of that
     * neighbor. 
     */
    static int receiveNeighborRank( const domain_type& domain, 
				    int neighbor_id )
    {
	return domain.receiveNeighborRank( neighbor_id );
    }

    /*!
     * \brief Get the number of neighbors to which this domain will send.
     */
    static int numSendNeighbors( const domain_type& domain )
    {
	return domain.numSendNeighbors();
    }

    /*!
     * \brief Given a local neighbor ID, return the proc rank of that
     * neighbor. 
     */
    static int sendNeighborRank( const domain_type& domain, int neighbor_id )
    {
	return domain.sendNeighborRank( neighbor_id );
    }

    /*!
     * \brief Given a state on the boundary or this domain, return the ID of
     * the owning neighbor.
     */
    static int owningNeighbor( const domain_type& domain, 
			       const ordinal_type state )
    {
	return domain.owningNeighbor( state );
    }
};

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

