//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in tally and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of tally code must retain the above copyright
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
 * \file MCLS_TallyTraits.hpp
 * \author Stuart R. Slattery
 * \brief Tally traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TALLYTRAITS_HPP
#define MCLS_TALLYTRAITS_HPP

#include "MCLS_RNGControl.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedTallyTraits
 * \brief Class for undefined tally traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Tally>
struct UndefinedTallyTraits
{
    static inline void notDefined()
    {
	return Tally::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class TallyTraits
 * \brief Traits for Monte Carlo transport tallies.
 *
 * TallyTraits defines an interface for parallel distributed tallies.
 */
template<class Tally>
class TallyTraits
{
  public:

    //@{
    //! Typedefs.
    typedef Tally                                      tally_type;
    typedef typename Tally::vector_type                vector_type;
    typedef typename Tally::ordinal_type               ordinal_type;
    typedef typename Tally::history_type               history_type;
    typedef Teuchos::Comm<int>                         Comm;
    //@}

    /*!
     * \brief Add a history's contribution to the tally.
     */
    static inline void tallyHistory( Tally& tally, 
				     history_type& history )
    { 
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Post-process a history after it has been killed permanently.
     */
    static inline void postProcessHistory( Tally& tally,
					   const history_type& history )
    { 
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Combine the tallies together over a set. This is generally
     * combining the tally and base tallies.
     */
    static void combineSetTallies( Tally& tally, 
				   const Teuchos::RCP<const Comm>& set_comm )
    {
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Combine the tallies together over a block communicator and
     * normalize by the number of sets (the size of a block).
     */
    static void combineBlockTallies( 
	Tally& tally,
	const Teuchos::RCP<const Comm>& block_comm,
        const int num_sets )
    {
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Normalize the tally with a specified number of histories.
     */
    static void normalize( Tally& tally, const int nh )
    {
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Set the tally base vector. The maps are required to be
     * compatible. 
     */
    static void setBaseVector( Tally& tally, 
			       const Teuchos::RCP<vector_type>& x_base )
    {
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Set the tallies to zero.
     */
    static void zeroOut( Tally& tally )
    {
	UndefinedTallyTraits<Tally>::notDefined(); 
    }

    /*!
     * \brief Get the estimator type used by this tally.
     */
    static int estimatorType( const Tally& tally )
    {
	UndefinedTallyTraits<Tally>::notDefined();
	return -1;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TALLYTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_TallyTraits.hpp
//---------------------------------------------------------------------------//

