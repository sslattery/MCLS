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
 * \file MCLS_AdjointHistory.hpp
 * \author Stuart R. Slattery
 * \brief Adjoint history class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTHISTORY_HPP
#define MCLS_ADJOINTHISTORY_HPP

#include <cmath>

#include "MCLS_HistoryTraits.hpp"
#include "MCLS_History.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class History
 * \brief Encapsulation of a random walk history's state for adjoint
 * calculations.
 */
//---------------------------------------------------------------------------//
template<class Ordinal>
class AdjointHistory : public History<Ordinal>
{
  public:

    //@{
    //! Typedefs.
    typedef Ordinal ordinal_type;
    typedef History<Ordinal> Base;
    //@}

    //! Default constructor.
    AdjointHistory()
    { /* ... */ }

    //! State constructor.
    AdjointHistory( Ordinal global_state, int local_state, double weight )
	: Base( global_state, local_state, weight )
    { /* ... */ }

    // Deserializer constructor.
    explicit AdjointHistory( const Teuchos::ArrayView<char>& buffer );

    // Pack the history into a buffer.
    Teuchos::Array<char> pack() const;

  public:

    // Set the byte size of the packed history state.
    static void setByteSize();

    // Get the number of bytes in the packed history state.
    static std::size_t getPackedBytes();

  private:

    // Packed size of history in bytes.
    static std::size_t d_packed_bytes;
};

//---------------------------------------------------------------------------//
// HistoryTraits Implementation.
//---------------------------------------------------------------------------//
template<class Ordinal>
class HistoryTraits<AdjointHistory<Ordinal> >
{
  public:

    //@{
    //! Typedefs.
    typedef AdjointHistory<Ordinal>                      history_type;
    typedef typename history_type::ordinal_type          ordinal_type;
    //@}

    /*!
     * \brief Create a history from a buffer.
     */
    static Teuchos::RCP<history_type> 
    createFromBuffer( const Teuchos::ArrayView<char>& buffer )
    { 
	return Teuchos::rcp( new history_type(buffer) );
    }

    /*!
     * \brief Pack the history into a buffer.
     */
    static Teuchos::Array<char> pack( const history_type& history )
    {
	return history.pack();
    }

    /*!
     * \brief Set the state of a history in global indexing.
     */
    static inline void setGlobalState( history_type& history, 
				       const ordinal_type state )
    {
	history.setGlobalState( state );
    }

    /*! 
     * \brief Get the state of a history in global indexing.
     */
    static inline ordinal_type globalState( const history_type& history )
    {
	return history.globalState();
    }

    /*!
     * \brief Set the state of a history in local indexing.
     */
    static inline void setLocalState( history_type& history, 
				      const int state )
    {
	history.setLocalState( state );
    }

    /*! 
     * \brief Get the state of a history in local indexing.
     */
    static inline int localState( const history_type& history )
    {
	return history.localState();
    }

    /*!
     * \brief Set the history weight.
     */
    static inline void setWeight( history_type& history, const double weight )
    {
	history.setWeight( weight );
    }

    /*! 
     * \brief Add to the history weight.
     */
    static inline void addWeight( history_type& history, const double weight )
    {
	history.addWeight( weight );
    }

    /*! 
     * \brief Multiply the history weight.
     */
    static inline void multiplyWeight( history_type& history, 
				       const double weight )
    {
	history.multiplyWeight( weight );
    }

    /*!
     * \brief Get the history weight.
     */
    static inline double weight( const history_type& history )
    {
	return history.weight();
    }

    /*!
     * \brief Get the absolute value of the history weight.
     */
    static inline double weightAbs( const history_type& history )
    {
	return history.weightAbs();
    }

    /*!
     * \brief Kill the history.
     */
    static inline void kill( history_type& history )
    {
	history.kill();
    }

    /*!
     * \brief Set the history alive
     */
    static inline void live( history_type& history )
    {
	history.live();
    }

    /*!
     * \brief Get the history live/dead status.
     */
    static inline bool alive( const history_type& history )
    {
	return history.alive();
    }

    /*!
     * \brief Set the event flag.
     */
    static inline void setEvent( history_type& history, const int event )
    {
	history.setEvent( event );
    }

    /*!
     * \brief Get the last event.
     */
    static inline int event( const history_type& history )
    {
	return history.event();
    }

    /*!
     * \brief Set the byte size of the packed history state.
     */
    static inline void setByteSize()
    {
	history_type::setByteSize();
    }

    /*!
     * \brief Get the number of bytes in the packed history state.
     */
    static inline std::size_t getPackedBytes()
    {
	return history_type::getPackedBytes();
    }

    /*!
     * \brief Add a step to the history.
     */
    static inline void addStep( history_type& history )
    {
	history.addStep();
    }

    /*!
     * \brief Get the number of steps this history has taken.
     */
    static inline int numSteps( const history_type& history )
    {
	return history.numSteps();
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AdjointHistory_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTHISTORY_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointHistory.hpp
//---------------------------------------------------------------------------//

