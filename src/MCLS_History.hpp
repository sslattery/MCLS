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
 * \brief History class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_HISTORY_HPP
#define MCLS_HISTORY_HPP

#include <cmath>

#include "MCLS_RNGControl.hpp"

#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class History
 * \brief Encapsulation of a random walk history's state.
 */
//---------------------------------------------------------------------------//
template<class Scalar, class Ordinal>
class History
{
  public:

    //@{
    //! Typedefs.
    typedef Scalar                                    scalar_type;
    typedef Ordinal                                   ordinal_type;
    typedef RNGControl::RNG                           RNG;
    //@}

    //! Default constructor.
    History()
	: d_state( Teuchos::OrdinalTraits<Ordinal>::zero() )
	, d_weight( Teuchos::ScalarTraits<Scalar>::one() )
	, d_alive( false )
    { /* ... */ }

    //! State constructor.
    History( Ordinal state, Scalar weight )
	: d_state( state )
	, d_weight( weight )
	, d_alive( false )
    { /* ... */ }

    // Deserializer constructor.
    explicit History( const Teuchos::ArrayView<char>& buffer );

    // Destructor.
    ~History()
    { /* ... */ }

    // Pack the history into a buffer.
    Teuchos::Array<char> pack() const;

    //! Set the history state.
    inline void setState( const Ordinal state )
    { d_state = state; }

    //! Get the history state.
    inline Ordinal state() const 
    { return d_state; }

    //! Set the history weight.
    inline void setWeight( const Scalar weight )
    { d_weight = weight; }

    //! Add to the history weight.
    inline void addWeight( const Scalar weight )
    { d_weight += weight; }

    //! Multiply the history weight.
    inline void multiplyWeight( const Scalar weight )
    { d_weight *= weight; }

    //! Get the history weight.
    inline Scalar weight() const
    { return d_weight; }

    //! Get the absolute value of the history weight.
    inline Scalar weightAbs() const
    { return std::abs(d_weight); }

    //! Set a new random number generator.
    void setRNG( const RNG& rng )
    { d_rng = rng; }

    //! Get the random number generator.
    const RNG& rng() const
    { return d_rng; }

    //! Kill the history.
    void kill()
    { d_alive = false; }

    //! Set the history alive.
    void live()
    { d_alive = true; }

    //! Get the history live/dead status.
    bool alive() const
    { return d_alive; }

    //! Set the event flag.
    void setEvent( int event )
    { d_event = event; }

    //! Get the last event.
    int event() const
    { return d_event; }

  public:

    // Set the byte size of the packed history state.
    static void setByteSize( std::size_t size_rng_state );

    // Get the number of bytes in the packed history state.
    static std::size_t getPackedBytes();

  private:

    //  history state.
    Ordinal d_state;

    // History weight.
    Scalar d_weight;

    // Random number generator.
    RNG d_rng;

    // Alive/dead status.
    bool d_alive;

    // Latest history event.
    int d_event;

  private:

    // Packed size of history in bytes.
    static std::size_t d_packed_bytes;

    // Packed size of the RNG in bytes.
    static std::size_t d_packed_rng;
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

