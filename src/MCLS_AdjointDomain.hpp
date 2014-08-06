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
 * \file MCLS_AdjointDomain.hpp
 * \author Stuart R. Slattery
 * \brief AdjointDomain declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTDOMAIN_HPP
#define MCLS_ADJOINTDOMAIN_HPP

#include <stack>
#include <set>
#include <unordered_map>
#include <random>

#include "MCLS_DBC.hpp"
#include "MCLS_DomainTraits.hpp"
#include "MCLS_AdjointTally.hpp"
#include "MCLS_SamplingTools.hpp"
#include "MCLS_Events.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_PRNG.hpp"
#include "MCLS_RNGTraits.hpp"
#include "MCLS_HistoryTraits.hpp"

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
 * \class AdjointDomain
 * \brief Adjoint transport domain.
 *
 * Derived from the adjoint Neumann-Ulam decomposition of a matrix.
 *
 * H^T = I - A^T 
 * H^T = (P) x (W)
 *
 * This domain contains data for all local states in the system, including the
 * overlap and neighboring domains. This object is responsible for creating
 * the tally for the solution vector over the domain as it has ownership of
 * the parallel decomposition of the domain.
 *
 * For all estimator types, the AdjointDomain constructs weights for all
 * transitions, cumulative distribution functions for each local state in the
 * system that can be sampled, and the associated states for those CDFs to
 * which a given initial state in the local system can transition to.
 *
 * If the expected value estimator is used, an additional copy of the
 * iteration matrix values are constructed. No other information is copied as
 * this data has the same non-zero structure and parallel distribution as the
 * CDFs generated for each state as the CDFs are derived from the iteration
 * matrix.
 */
template<class Vector, class Matrix, class RNG>
class AdjointDomain
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                        vector_type;
    typedef VectorTraits<Vector>                          VT;
    typedef Matrix                                        matrix_type;
    typedef MatrixTraits<Vector,Matrix>                   MT;
    typedef typename VT::global_ordinal_type              Ordinal;
    typedef AdjointTally<Vector>                          TallyType;
    typedef typename TallyType::HistoryType               HistoryType;
    typedef HistoryTraits<HistoryType>                    HT;
    typedef std::stack<Teuchos::RCP<HistoryType> >        BankType;
    typedef typename std::unordered_map<Ordinal,int>      MapType;
    typedef Teuchos::Comm<int>                            Comm;
    typedef RNG                                           rng_type;
    typedef RNGTraits<RNG>                                RNGT;
    typedef typename RNGT::uniform_real_distribution_type RandomDistribution;
    typedef RandomDistributionTraits<RandomDistribution>  RDT;
    //@}

    // Matrix constructor.
    AdjointDomain( const Teuchos::RCP<const Matrix>& A,
		   const Teuchos::RCP<Vector>& x,
		   const Teuchos::ParameterList& plist );

    // Deserializer constructor.
    AdjointDomain( const Teuchos::ArrayView<char>& buffer,
		   const Teuchos::RCP<const Comm>& set_comm );

    // Destructor.
    ~AdjointDomain()
    { /* ... */ }

    // Set the random number generator.
    void setRNG( const Teuchos::RCP<PRNG<RNG> >& rng )
    { d_rng = rng; }

    // Pack the domain into a buffer.
    Teuchos::Array<char> pack() const;

    // Get the size of this object in packed bytes.
    std::size_t getPackedBytes() const;

    //! Set the weight cutoff.
    void setCutoff( const double weight_cutoff )
    { d_weight_cutoff = weight_cutoff; }

    // Process a history through a transition to a new state.
    inline void processTransition( HistoryType& history );

    //! Deterimine if a history should be terminated.
    inline bool terminateHistory( const HistoryType& history )
    { return HT::weightAbs(history) < d_weight_cutoff; }

    // Get the domain tally.
    Teuchos::RCP<TallyType> domainTally() const
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

  private:

    // Add matrix data to the local domain.
    void addMatrixToDomain( const Teuchos::RCP<const Matrix>& A,
                            std::set<Ordinal>& tally_states,
			    const double relaxation );

    // Build boundary data.
    void buildBoundary( const Teuchos::RCP<const Matrix>& A,
			const Teuchos::RCP<const Matrix>& base_A );

  private:

    // Random number generator.
    Teuchos::RCP<PRNG<RNG> > d_rng;

    // Random number distribution.
    Teuchos::RCP<RandomDistribution> d_rng_dist;

    // Monte Carlo estimator type.
    int d_estimator;

    // History weight cutoff.
    double d_weight_cutoff;

    // Domain tally.
    Teuchos::RCP<TallyType> d_tally;

    // Global-to-local row indexer.
    Teuchos::RCP<MapType> d_g2l_row_indexer;

    // Local CDF columns.
    Teuchos::ArrayRCP<Teuchos::RCP<Teuchos::Array<Ordinal> > > d_columns;

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
    MapType d_bnd_to_neighbor;
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Process a history through a transition to a new state.
 */
template<class Vector, class Matrix, class RNG>
inline void AdjointDomain<Vector,Matrix,RNG>::processTransition( 
    HistoryType& history )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_rng) );
    MCLS_REQUIRE( HT::alive(history) );
    MCLS_REQUIRE( Event::TRANSITION == HT::event(history) );
    MCLS_REQUIRE( isGlobalState(HT::globalState(history)) );

    // Get the current state.
    typename MapType::const_iterator index = 
	d_g2l_row_indexer->find( HT::globalState(history) );
    MCLS_CHECK( index != d_g2l_row_indexer->end() );

    // Sample the row CDF to get a new state.
    Ordinal new_state = 
	SamplingTools::sampleDiscreteCDF( d_cdfs[index->second](),
					  d_rng->random(*d_rng_dist) );
    HT::setGlobalState( history, (*d_columns[index->second])[new_state] );

    // Update the history weight with the transition weight.
    HT::multiplyWeight( history,
			d_weights[index->second] *
			d_h[index->second][new_state] /
			std::abs(d_h[index->second][new_state]) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given global state is in the local domain.
 */
template<class Vector, class Matrix, class RNG>
inline bool AdjointDomain<Vector,Matrix,RNG>::isGlobalState( 
    const Ordinal& state ) const
{
    return ( d_g2l_row_indexer->end() != d_g2l_row_indexer->find(state) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given global state is on the boundary.
 */
template<class Vector, class Matrix, class RNG>
inline bool AdjointDomain<Vector,Matrix,RNG>::isBoundaryState( 
    const Ordinal& state ) const
{
    return ( d_bnd_to_neighbor.end() != d_bnd_to_neighbor.find(state) );
}

//---------------------------------------------------------------------------//
// DomainTraits implementation.
//---------------------------------------------------------------------------//
/*!
 * \class DomainTraits
 * \brief Traits implementation for the AdjointDomain.
 */
template<class Vector, class Matrix, class RNG>
class DomainTraits<AdjointDomain<Vector,Matrix,RNG> >
{
  public:

    //@{
    //! Typedefs.
    typedef AdjointDomain<Vector,Matrix,RNG>            domain_type;
    typedef typename domain_type::Ordinal               ordinal_type;
    typedef typename domain_type::HistoryType           history_type;
    typedef typename domain_type::TallyType             tally_type;
    typedef typename domain_type::BankType              bank_type;
    typedef typename domain_type::rng_type              rng_type;
    typedef Teuchos::Comm<int>                          Comm;
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
     * \brief Create a reference-counted pointer to a new domain defined over
     * the given communicator by unpacking a data buffer. 
     */
    static Teuchos::RCP<domain_type> 
    createFromBuffer( 
	const Teuchos::RCP<const Comm>& comm,
	const Teuchos::ArrayView<char>& buffer )
    { 
	return Teuchos::rcp( new domain_type(buffer,comm) );
    }

    /*!
     * \brief Pack a domain into a buffer.
     */
    static Teuchos::Array<char> pack( const domain_type& domain )
    { 
	return domain.pack();
    }

    /*!
     * \brief Get the size of domain in packed bytes.
     */
    static std::size_t getPackedBytes( const domain_type& domain )
    { 
	return domain.getPackedBytes();
    }

    /*!
     * \brief Set a weight cutoff with the domain.
     */
    static void setCutoff( domain_type& domain, const double cutoff )
    { 
	domain.setCutoff( cutoff );
    }

    /*!
     * \brief Process a history through a transition in the local domain to a
     * new state
     */
    static inline void processTransition( 
	domain_type& domain, history_type& history )
    { 
	domain.processTransition( history );
    }

    /*!
     * \brief Deterimine if a history should be terminated.
     */
    static inline bool terminateHistory( 
	domain_type& domain, const history_type& history )
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
     * \brief Determine if a given global state is in the local domain.
     */
    static bool isGlobalState( const domain_type& domain, 
			      const ordinal_type state )
    { 
	return domain.isGlobalState( state );
    }

    /*!
     * \brief Determine if a given state is on the boundary.
     */
    static bool isBoundaryState( const domain_type& domain, 
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

#include "MCLS_AdjointDomain_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTDOMAIN_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointDomain.hpp
// ---------------------------------------------------------------------------//

