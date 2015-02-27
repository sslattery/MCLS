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
 * \file MCLS_ForwardDomain.hpp
 * \author Stuart R. Slattery
 * \brief ForwardDomain declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FORWARDDOMAIN_HPP
#define MCLS_FORWARDDOMAIN_HPP

#include <stack>
#include <unordered_map>
#include <set>
#include <random>

#include "MCLS_DBC.hpp"
#include "MCLS_DomainTraits.hpp"
#include "MCLS_AlmostOptimalDomain.hpp"
#include "MCLS_HistoryTraits.hpp"
#include "MCLS_ForwardTally.hpp"
#include "MCLS_SamplingTools.hpp"
#include "MCLS_Events.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_PRNG.hpp"
#include "MCLS_RNGTraits.hpp"

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
 * \class ForwardDomain
 * \brief Forward transport domain.
 *
 * Derived from the forward Neumann-Ulam product of a matrix.
 *
 * H = I - A 
 * H = (P) x (W)
 *
 * This domain contains data for all local states in the system, including the
 * overlap and neighboring domains. This object is responsible for creating
 * the tally for the solution vector over the domain as it has ownership of
 * the parallel decomposition of the domain.
 *
 * For all estimator types, the ForwardDomain constructs weights for all
 * transitions, cumulative distribution functions for each local state in the
 * system that can be sampled, and the associated states for those CDFs to
 * which a given initial state in the local system can transition to.
 *
 * If the expected value estimator is used, and additional copy of the
 * iteration matrix values are constructed. No other information is copied as
 * this data has the same non-zero structure and parallel distribution as the
 * CDFs generated for each state as the CDFs are derived from the iteration
 * matrix.
 *
 * The expected value estimator is currently not supported for forward
 * problems. 
 */
template<class Vector, class Matrix, class RNG>
class ForwardDomain : 
	public AlmostOptimalDomain<Vector,Matrix,RNG,ForwardTally<Vector> >
{
  public:

    //@{
    //! Typedefs.
    typedef AlmostOptimalDomain<Vector,Matrix,RNG,ForwardTally<Vector> > Base;
    typedef Vector                                        vector_type;
    typedef VectorTraits<Vector>                          VT;
    typedef Matrix                                        matrix_type;
    typedef MatrixTraits<Vector,Matrix>                   MT;
    typedef typename VT::global_ordinal_type              Ordinal;
    typedef ForwardTally<Vector>                          TallyType;
    typedef typename TallyType::HistoryType               HistoryType;
    typedef HistoryTraits<HistoryType>                    HT;
    typedef std::stack<Teuchos::RCP<HistoryType> >        BankType;
    typedef Teuchos::Comm<int>                            Comm;
    typedef RNG                                           rng_type;
    typedef RNGTraits<RNG>                                RNGT;
    //@}

    // Matrix constructor.
    ForwardDomain( const Teuchos::RCP<const Matrix>& A,
		   const Teuchos::RCP<Vector>& x,
		   const Teuchos::ParameterList& plist );

    // Deserializer constructor.
    ForwardDomain( const Teuchos::ArrayView<char>& buffer,
		   const Teuchos::RCP<const Comm>& set_comm );

    // Destructor.
    ~ForwardDomain()
    { /* ... */ }

    // Pack the domain into a buffer.
    Teuchos::Array<char> pack() const;

    // Get the size of this object in packed bytes.
    std::size_t getPackedBytes() const;
};

//---------------------------------------------------------------------------//
// DomainTraits implementation.
//---------------------------------------------------------------------------//
/*!
 * \class DomainTraits
 * \brief Traits implementation for the ForwardDomain.
 */
template<class Vector, class Matrix, class RNG>
class DomainTraits<ForwardDomain<Vector,Matrix,RNG> >
{
  public:

    //@{
    //! Typedefs.
    typedef ForwardDomain<Vector,Matrix,RNG>            domain_type;
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

#include "MCLS_ForwardDomain_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_FORWARDDOMAIN_HPP

//---------------------------------------------------------------------------//
// end MCLS_ForwardDomain.hpp
// ---------------------------------------------------------------------------//

