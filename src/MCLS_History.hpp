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

#include <Teuchos_RCP.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_OrdinalTraits.hpp>

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
    //@}

    //! Default constructor.
    History()
	: d_weight( Teuchos::ScalarTraits<Scalar>::one() )
	, d_state( Teuchos::OrdinalTraits<Ordinal>::zero() )
    { /* ... */ }

    //! State constructor.
    */
    History( Scalar weight, Ordinal state )
	: d_weight( weight )
	, d_state( state )
    { /* ... */ }

    // Destructor.
    ~History()
    { /* ... */ }

    //! Set the history weight.
    inline void setWeight( const Scalar weight )
    { d_weight = weight; }

    //! Add to the history weight.
    inline void addWeight( const Scalar weight )
    { d_weight += weight; }

    //! Multiply the history weight.
    inline void multiplyWeight( const Scalar weight )
    { d_weight *= weight; }

    //! Set the history state.
    inline void setState( const Ordinal state )
    { d_state = state; }

    //! Get the history weight.
    inline Scalar weight() const
    { return d_weight; }

    //! Get the absolute value of the history weight.
    inline Scalar weightAbs() const
    { return std::abs(d_weight); }

    //! Get the history state.
    inline Ordinal State() const 
    { return d_state; }

  private:

    // History weight.
    Scalar d_weight;

    //  history state.
    Ordinal d_state;

    // Random number generator (reference counted).
    RNG d_rng;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_HISTORY_HPP

//---------------------------------------------------------------------------//
// end MCLS_History.hpp
//---------------------------------------------------------------------------//

