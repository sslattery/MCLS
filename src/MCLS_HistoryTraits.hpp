//---------------------------------------------------------------------------//
/*
  Copyright (c) 2012, Stuart R. Slattery
  All rights reserved.

  Redistribution and use in history and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

  *: Redistributions of history code must retain the above copyright
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
 * \file MCLS_HistoryTraits.hpp
 * \author Stuart R. Slattery
 * \brief History traits definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_HISTORYTRAITS_HPP
#define MCLS_HISTORYTRAITS_HPP

#include "MCLS_RNGControl.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedHistoryTraits
 * \brief Class for undefined history traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class History>
struct UndefinedHistoryTraits
{
    static inline void notDefined()
    {
	return History::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class HistoryTraits
 * \brief Traits for Monte Carlo transport histories.
 *
 * HistoryTraits defines an interface for stochastic histories.
 */
template<class History>
class HistoryTraits
{
  public:

    //@{
    //! Typedefs.
    typedef History                                      history_type;
    typedef typename History::ordinal_type               ordinal_type;
    typedef typename History::RNG                        rng_type;
    //@}

    /*!
     * \brief Create a history from a buffer.
     */
    static Teuchos::RCP<history_type> 
    createFromBuffer( const Teuchos::ArrayView<char>& buffer )
    { 
	UndefinedHistoryTraits<History>::notDefined(); 
	return Teuchos::null;
    }

    /*!
     * \brief Pack the history into a buffer.
     */
    static Teuchos::Array<char> pack( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
	return Teuchos::Array<char>(0);
    }

    /*!
     * \brief Set the state of a history
     */
    static inline void setState( history_type& history, 
				 const ordinal_type state )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*! 
     * \brief get the state of a history.
     */
    static inline ordinal_type state( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
	return -1;
    }

    /*!
     * \brief Set the history weight.
     */
    static inline void setWeight( history_type& history, const double weight )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*! 
     * \brief Add to the history weight.
     */
    static inline void addWeight( history_type& history, const double weight )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*! 
     * \brief Multiply the history weight.
     */
    static inline void multiplyWeight( history_type& history, 
				       const double weight )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*!
     * \brief Get the history weight.
     */
    static inline double weight( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
	return 0.0;
    }

    /*!
     * \brief Get the absolute value of the history weight.
     */
    static inline double weightAbs( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
	return -1.0;
    }

    /*!
     * \brief Set a new random number generator with the history.
     */
    static void setRNG( history_type& history, const rng_type& rng )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*!
     * \brief Get this history's random number generator.
     */
    static const rng_type& rng( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*!
     * \brief Kill the history.
     */
    static void kill( history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*!
     * \brief Set the history alive
     */
    static void live( history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*!
     * \brief Get the history live/dead status.
     */
    static bool alive( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
	return false;
    }

    /*!
     * \brief Set the event flag.
     */
    static void setEvent( history_type& history, const int event )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
    }

    /*!
     * \brief Get the last event.
     */
    static int event( const history_type& history )
    {
	UndefinedHistoryTraits<History>::notDefined(); 
	return -1;
    }

    /*!
     * \brief Set the byte size of the packed history state.
     */
    static void setByteSize( std::size_t size_rng_state )
    {
	UndefinedHistoryTraits<History>::notDefined();
    }

    /*!
     * \brief Get the number of bytes in the packed history state.
     */
    static std::size_t getPackedBytes()
    {
	UndefinedHistoryTraits<History>::notDefined();
	return 0;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_HISTORYTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_HistoryTraits.hpp
//---------------------------------------------------------------------------//

