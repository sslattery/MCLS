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

#include "MCLS_RNGControl.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class Source 
 * \brief Structure for communicating histories amongst domains in a set. 
 *
 * Tom Evans is responsible for the design of this class.
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
    typedef typename Domain::VectorType                  VectorType;
    typedef Teuchos::Comm<int>                           Comm;
    //@}
  public:

    // Constructor.
    Source( const Teuchos::RCP<VectorType>& b,
	    const Teuchos::RCP<Domain>& domain,
	    const Teuchos::RCP<const Comm>& set_const_comm,
	    const Teuchos::ParameterList& plist );

    // Destructor.
    ~Source()
    { /* ... */ }

  private:

    // Source vector.
    Teuchos::RCP<VectorType> d_b;

    // Local domain.
    Teuchos::RCP<Domain> d_domain;

    // Set-constant communicator for domain-to-domain communcation.
    Teuchos::RCP<const Comm> d_comm;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_Source_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_SOURCE_HPP

//---------------------------------------------------------------------------//
// end MCLS_Source.hpp
//---------------------------------------------------------------------------//

