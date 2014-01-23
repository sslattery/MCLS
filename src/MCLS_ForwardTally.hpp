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
 * \file MCLS_ForwardTally.hpp
 * \author Stuart R. Slattery
 * \brief ForwardTally declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FORWARDTALLY_HPP
#define MCLS_FORWARDTALLY_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_ForwardHistory.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_TallyTraits.hpp"
#include "MCLS_Estimators.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>

#include <boost/tr1/unordered_map.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class ForwardTally
 * \brief Monte Carlo tally for the linear system solution vector for forward
 * problems. 
 */
template<class Vector>
class ForwardTally
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                              vector_type;
    typedef VectorTraits<Vector>                                VT;
    typedef typename VT::global_ordinal_type                    Ordinal;
    typedef typename VT::scalar_type                            Scalar;
    typedef ForwardHistory<Ordinal>                             HistoryType;
    typedef Teuchos::Comm<int>                                  Comm;
    typedef typename std::tr1::unordered_map<Ordinal,int>       MapType;
    //@}

    // Constructor.
    ForwardTally( const Teuchos::RCP<Vector>& x,
                  const int estimator );

    // Destructor.
    ~ForwardTally()
    { /* ... */ }

    // Assign the source vector to the tally.
    void setSource( const Teuchos::RCP<Vector>& b );

    // Add a history's contribution to the tally.
    inline void tallyHistory( HistoryType& history );

    // Post-process a history if it is permanently killed in the local domain.
    void postProcessHistory( const HistoryType& history );

    // Combine the overlap tally with the base decomposition tally in the set
    // and normalize by the counted number of histories in each set.
    void combineSetTallies( const Teuchos::RCP<const Comm>& set_comm );

    // Combine the secondary tallies with the primary tally across a
    // block. Normalize the result with the number of sets.
    void combineBlockTallies( const Teuchos::RCP<const Comm>& block_comm,
                              const int num_sets );

    // Normalize base decomposition tally with the number of specified
    // histories.
    void normalize( const int& nh );

    // Set the base tally vector.
    void setBaseVector( const Teuchos::RCP<Vector>& x_base );

    // Zero out the tallies.
    void zeroOut();

    // Get the number global rows in the base decomposition.
    Ordinal numBaseRows() const;

    // Get the global tally rows in the base decomposition.
    Teuchos::Array<Ordinal> baseRows() const;

    //! Get the estimator type for this tally.
    int estimatorType() const { return d_estimator; }

  private:

    // Solution vector in operator decomposition.
    Teuchos::RCP<Vector> d_x;

    // Source vector in operator decomposition.
    Teuchos::RCP<Vector> d_b;

    // Tally states.
    Teuchos::Array<Ordinal> d_tally_states;

    // Tally values.
    Teuchos::Array<Scalar> d_tally_values;

    // Tally count.
    Teuchos::Array<int> d_tally_count;

    // Monte Carlo estimator type.
    int d_estimator;
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*
 * \brief Add a history's contribution to the tally.
 */
template<class Vector>
inline void ForwardTally<Vector>::tallyHistory( HistoryType& history )
{
    MCLS_REQUIRE( history.alive() );
    MCLS_REQUIRE( Teuchos::nonnull(d_b) );
    MCLS_REQUIRE( VT::isGlobalRow(*d_b,history.state()) );
    MCLS_REQUIRE( Estimator::COLLISION == d_estimator );

    typename VT::local_ordinal_type local_state = 
	VT::getLocalRow( *d_b, history.state() );
    history.addToHistoryTally( history.weight() * VT::view(*d_b)[local_state] );
}

//---------------------------------------------------------------------------//
// TallyTraits implementation.
//---------------------------------------------------------------------------//
/*!
 * \class TallyTraits
 * \brief Specialization for ForwardTally.
 */
template<class Vector>
class TallyTraits<ForwardTally<Vector> >
{
  public:

    //@{
    //! Typedefs.
    typedef ForwardTally<Vector>                       tally_type;
    typedef typename tally_type::vector_type           vector_type;
    typedef typename tally_type::Ordinal               ordinal_type;
    typedef typename tally_type::HistoryType           history_type;
    typedef Teuchos::Comm<int>                         Comm;
    //@}

    /*!
     * \brief Add a history's contribution to the tally.
     */
    static inline void tallyHistory( tally_type& tally, 
				     history_type& history )
    { 
	tally.tallyHistory( history );
    }

    /*!
     * \brief Post-process a history after it has been killed permanently.
     */
    static inline void postProcessHistory( tally_type& tally,
					   const history_type& history )
    { 
	tally.postProcessHistory( history );
    }

    /*!
     * \brief Combine the tallies together over a set. This is generally
     * combining the overlap and base tallies.
     */
    static void combineSetTallies( tally_type& tally,
				   const Teuchos::RCP<const Comm>& set_comm )
    {
	tally.combineSetTallies( set_comm );
    }

    /*!
     * \brief Combine the tallies together over a block communicator.
     */
    static void combineBlockTallies( 
	tally_type& tally,
	const Teuchos::RCP<const Comm>& block_comm,
        const int num_sets )
    {
	tally.combineBlockTallies( block_comm, num_sets );
    }

    /*!
     * \brief Normalize the tally with a specified number of histories.
     */
    static void normalize( tally_type& tally, const int nh )
    {
	tally.normalize( nh );
    }

    /*!
     * \brief Set the tally base vector. The maps are required to be
     * compatible. 
     */
    static void setBaseVector( tally_type& tally, 
			       const Teuchos::RCP<vector_type>& x_base )
    {
	tally.setBaseVector( x_base );
    }

    /*!
     * \brief Set the tallies to zero.
     */
    static void zeroOut( tally_type& tally )
    {
	tally.zeroOut();
    }

    /*!
     * \brief Get the estimator type used by this tally.
     */
    static int estimatorType( const tally_type& tally )
    {
	return tally.estimatorType();
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_ForwardTally_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_FORWARDTALLY_HPP

//---------------------------------------------------------------------------//
// end MCLS_ForwardTally.hpp
// ---------------------------------------------------------------------------//

