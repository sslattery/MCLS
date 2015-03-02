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
 * \file MCLS_DomainTraits.hpp
 * \author Stuart R. Slattery
 * \brief Domain traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_DOMAINTRAITS_HPP
#define MCLS_DOMAINTRAITS_HPP

#include "MCLS_PRNG.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedDomainTraits
 * \brief Class for undefined domain traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Domain>
struct UndefinedDomainTraits
{
    static inline void notDefined()
    {
	return Domain::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class DomainTraits
 * \brief Traits for Monte Carlo transport domains.
 *
 * DomainTraits defines an interface for parallel distributed domains.
 */
template<class Domain>
class DomainTraits
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                      domain_type;
    typedef typename Domain::ordinal_type               ordinal_type;
    typedef typename Domain::history_type               history_type;
    typedef typename Domain::tally_type                 tally_type;
    typedef typename Domain::bank_type                  bank_type;
    typedef typename Domain::rng_type                   rng_type;
    typedef Teuchos::Comm<int>                          Comm;
    //@}

    /*!
     * \brief Set a random number generator with the domain.
     */
    static void setRNG( Domain& domain,
			const Teuchos::RCP<PRNG<rng_type> >& rng )
    {
	UndefinedDomainTraits<Domain>::notDefined(); 
    }

    /*!
     * \brief Given a history with a global state in the local domain, set the
     * local state of that history.
     */
    static inline void setHistoryLocalState( 
	const Domain& domain, history_type& history )
    { 
	UndefinedDomainTraits<Domain>::notDefined(); 
    }

    /*!
     * \brief Process a history through a transition in the local domain to a
     * new state.
     */
    static inline void processTransition( 
	const Domain& domain, history_type& history )
    { 
	UndefinedDomainTraits<Domain>::notDefined(); 
    }

    /*!
     * \brief Deterimine if a history should be terminated.
     */
    static inline bool terminateHistory( 
	const Domain& domain, const history_type& history )
    { 
	UndefinedDomainTraits<Domain>::notDefined(); 
	return false;
    }

    /*!
     * \brief Get the tally associated with this domain.
     */
    static Teuchos::RCP<tally_type> domainTally( const Domain& domain )
    { 
	UndefinedDomainTraits<Domain>::notDefined(); 
	return Teuchos::null;
    }

    /*!
     * \brief Determine if a given state is in the local domain.
     */
    static inline bool isGlobalState( const Domain& domain, const ordinal_type state )
    { 
	UndefinedDomainTraits<Domain>::notDefined(); 
	return false;
    }

    /*!
     * \brief Determine if a given state is on the boundary.
     */
    static inline bool isBoundaryState( const Domain& domain, const ordinal_type state )
    { 
	UndefinedDomainTraits<Domain>::notDefined(); 
	return false;
    }

    /*!
     * \brief Get the number of neighbors from which this domain will
     * receive. 
     */
    static int numReceiveNeighbors( const Domain& domain )
    {
	UndefinedDomainTraits<Domain>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Given a local neighbor ID, return the proc rank of that
     * neighbor. 
     */
    static int receiveNeighborRank( const Domain& domain, int neighbor_id )
    {
	UndefinedDomainTraits<Domain>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Get the number of neighbors to which this domain will send.
     */
    static int numSendNeighbors( const Domain& domain )
    {
	UndefinedDomainTraits<Domain>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Given a local neighbor ID, return the proc rank of that
     * neighbor. 
     */
    static int sendNeighborRank( const Domain& domain, int neighbor_id )
    {
	UndefinedDomainTraits<Domain>::notDefined(); 
	return 0;
    }

    /*!
     * \brief Given a state on the boundary or this domain, return the ID of
     * the owning neighbor.
     */
    static int owningNeighbor( const Domain& domain, const ordinal_type state )
    {
	UndefinedDomainTraits<Domain>::notDefined(); 
	return 0;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_DOMAINTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_DomainTraits.hpp
//---------------------------------------------------------------------------//

