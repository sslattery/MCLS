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

#include <unordered_map>

#include "MCLS_DBC.hpp"
#include "MCLS_ForwardHistory.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_TallyTraits.hpp"

#include <Teuchos_RCP.hpp>

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
    //@}

    // Constructor.
    ForwardTally( const Teuchos::RCP<Vector>& x );

    // Get the vector under the tally.
    Teuchos::RCP<Vector> getVector() const
    { return d_x; }

    // Assign the source vector to the tally.
    void setSource( const Teuchos::RCP<Vector>& b );

    // Add a history's contribution to the tally.
    inline void tallyHistory( HistoryType& history );

    // Post-process a history if it is permanently killed in the local domain.
    void postProcessHistory( const HistoryType& history );

    // Normalize base decomposition tally with the number of specified
    // histories.
    void normalize( const int& nh );

    // Zero out the tallies.
    void zeroOut();

    // Finalize the tally.
    void finalize();

  private:

    // Solution vector in operator decomposition.
    Teuchos::RCP<Vector> d_x;

    // Source vector in operator decomposition.
    Teuchos::RCP<Vector> d_b;

    // View of the local source.
    Teuchos::ArrayRCP<const Scalar> d_b_view;

    // Tally states, values, and counts.
    std::unordered_map<Ordinal,std::pair<Scalar,int> > d_states_values_counts;
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
    MCLS_REQUIRE( VT::isLocalRow(*d_b,history.localState()) );

    history.addToHistoryTally( 
	history.weight() * d_b_view[history.localState()] );
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
    //@}

    /*!
     * \brief Factory method. This methods builds a tally around a vector.
     */
    static Teuchos::RCP<tally_type>
    create( const Teuchos::RCP<vector_type>& vector )
    { 
	return Teuchos::rcp( new tally_type(vector) );
    }

    /*!
     * \brief Get the vector under the tally.
     */
    static Teuchos::RCP<vector_type> getVector( const tally_type& tally )
    {
	return tally.getVector();
    }

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
     * \brief Normalize the tally with a specified number of histories.
     */
    static void normalize( tally_type& tally, const int nh )
    {
	tally.normalize( nh );
    }

    /*!
     * \brief Set the tallies to zero.
     */
    static void zeroOut( tally_type& tally )
    {
	tally.zeroOut();
    }

    /*!
     * \brief Finalize the tally.
     */
    static void finalize( tally_type& tally )
    {
	tally.finalize();
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

