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
 * \file MCLS_Source.hpp
 * \author Stuart R. Slattery
 * \brief Source class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOURCE_HPP
#define MCLS_SOURCE_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_RNGControl.hpp"

#include <Teuchos_RCP.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class Source 
 * \brief Base class for Monte Carlo history sources.
 *
 * This class and inheritance structure is based on that developed by Tom
 * Evans. 
 */
//---------------------------------------------------------------------------//
template<class Domain>
class Source
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                       domain_type;
    typedef typename Domain::HistoryType                 HistoryType;
    typedef typename Domain::vector_type                 VectorType;
    typedef RNGControl::RNG                              RNG;
    //@}

    // No source constructor.
    Source( const Teuchos::RCP<Domain>& domain,
	    const Teuchos::RCP<RNGControl>& rng_control );


    // Source constructor.
    Source( const Teuchos::RCP<VectorType>& b,
	    const Teuchos::RCP<Domain>& domain,
	    const Teuchos::RCP<RNGControl>& rng_control );

    // Destructor.
    virtual ~Source() = 0;

    //! Get a history from the source.
    virtual Teuchos::RCP<HistoryType> getHistory() = 0;

    //! Return whether the source has emitted all histories.
    virtual bool empty() const = 0;

    //! Get the number of source histories to transport in the local domain.
    virtual int numToTransport() const = 0;

    //! Get the number of source histories in the set.
    virtual int numToTransportInSet() const = 0;

    // Set the source vector.
    void setSourceVector( const Teuchos::RCP<VectorType>& b );

    //! Get the source vector.
    const VectorType& sourceVector() const { return *b_b; }

    //! Get the domain.
    const Domain& domain() const { return *b_domain; }

    // Get the RNG controller.
    const RNGControl& rngControl() const { return *b_rng_control; }

  protected:

    // Source vector.
    Teuchos::RCP<VectorType> b_b;

    // Local domain.
    Teuchos::RCP<Domain> b_domain;

    // Random number controller.
    Teuchos::RCP<RNGControl> b_rng_control;
};

//---------------------------------------------------------------------------//
// Implementation.
//---------------------------------------------------------------------------//
/*!
 * \brief No source constructor.
 */
template<class Domain>
Source<Domain>::Source( const Teuchos::RCP<Domain>& domain,
			const Teuchos::RCP<RNGControl>& rng_control )
    : b_domain( domain )
    , b_rng_control( rng_control )
{
    Require( !b_domain.is_null() );
    Require( !b_rng_control.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Domain>
Source<Domain>::Source( const Teuchos::RCP<VectorType>& b,
			const Teuchos::RCP<Domain>& domain,
			const Teuchos::RCP<RNGControl>& rng_control )
    : b_b( b )
    , b_domain( domain )
    , b_rng_control( rng_control )
{
    Require( !b_b.is_null() );
    Require( !b_domain.is_null() );
    Require( !b_rng_control.is_null() );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor. Pure virtual to prohibit direct generation of this
 * class. 
 */
template<class Domain>
Source<Domain>::~Source()
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Set the source vector.
 */
template<class Domain>
void Source<Domain>::setSourceVector( const Teuchos::RCP<VectorType>& b)
{
    Require( !b.is_null() );

    b_b = b;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_SOURCE_HPP

//---------------------------------------------------------------------------//
// end MCLS_Source.hpp
//---------------------------------------------------------------------------//

