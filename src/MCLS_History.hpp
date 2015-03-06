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
 * \file MCLS_History.hpp
 * \author Stuart R. Slattery
 * \brief History base class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_HISTORY_HPP
#define MCLS_HISTORY_HPP

#include <cmath>

#include "MCLS_Serializer.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class History
 * \brief Base class for encapsulation of a random walk history's state.
 */
//---------------------------------------------------------------------------//
template<class Ordinal>
class History
{
  public:

    //@{
    //! Typedefs.
    typedef Ordinal ordinal_type;
    //@}

    //! Default constructor.
    History()
	: b_global_state( Teuchos::OrdinalTraits<Ordinal>::invalid() )
	, b_local_state( Teuchos::OrdinalTraits<Ordinal>::invalid() )
	, b_weight( Teuchos::ScalarTraits<double>::one() )
	, b_alive( false )
	, b_event( 0 )
	, b_num_steps( 0 )
    { /* ... */ }

    //! State constructor.
    History( Ordinal global_state, int local_state, double weight )
	: b_global_state( global_state )
	, b_local_state( local_state )
	, b_weight( weight )
	, b_alive( false )
	, b_event( 0 )
	, b_num_steps( 0 )
    { /* ... */ }

    // Pack the history into a buffer.
    void packHistory( Serializer& s ) const;

    // Unpack the history from a buffer.
    void unpackHistory( Deserializer& ds );

    //! Set the history state in global indexing.
    inline void setGlobalState( const Ordinal global_state )
    { b_global_state = global_state; }

    //! Get the history state in global indexing.
    inline Ordinal globalState() const 
    { return b_global_state; }

    //! Set the history state in local indexing.
    inline void setLocalState( const int local_state )
    { b_local_state = local_state; }

    //! Get the history state in local indexing.
    inline int localState() const 
    { return b_local_state; }

    //! Set the history weight.
    inline void setWeight( const double weight )
    { b_weight = weight; }

    //! Add to the history weight.
    inline void addWeight( const double weight )
    { b_weight += weight; }

    //! Multiply the history weight.
    inline void multiplyWeight( const double weight )
    { b_weight *= weight; }

    //! Get the history weight.
    inline double weight() const
    { return b_weight; }

    //! Get the absolute value of the history weight.
    inline double weightAbs() const
    { return std::abs(b_weight); }

    //! Kill the history.
    inline void kill()
    { b_alive = false; }

    //! Set the history alive.
    inline void live()
    { b_alive = true; }

    //! Get the history live/dead status.
    inline bool alive() const
    { return b_alive; }

    //! Set the event flag.
    inline void setEvent( const int event )
    { b_event = event; }

    //! Get the last event.
    inline int event() const
    { return b_event; }

    //! Add a step to the history.
    inline void addStep()
    { ++b_num_steps; }
    
    //! Get the number of steps this history has taken.
    inline int numSteps() const
    { return b_num_steps; }
    
  public:

    // Set the byte size of the packed history state.
    static void setStaticSize();

    // Get the number of bytes in the packed history state.
    static std::size_t getStaticSize();

  protected:

    // History state in global indexing.
    Ordinal b_global_state;

    // History state in local indexing.
    int b_local_state;

    // History weight.
    double b_weight;

    // Alive/dead status.
    bool b_alive;

    // Latest history event.
    int b_event;

    // Number of steps this history has taken.
    int b_num_steps;
    
  private:

    // Packed size of history in bytes.
    static std::size_t b_packed_bytes;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_History_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_HISTORY_HPP

//---------------------------------------------------------------------------//
// end MCLS_History.hpp
//---------------------------------------------------------------------------//

