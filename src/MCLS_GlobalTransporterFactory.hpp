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
 * \file MCLS_GlobalTransporterFactory.hpp
 * \author Stuart R. Slattery
 * \brief GlobalTransporterFactory class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_GLOBALTRANSPORTERFACTORY_HPP
#define MCLS_GLOBALTRANSPORTERFACTORY_HPP

#include "MCLS_GlobalTransporter.hpp"
#include "MCLS_SourceTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class GlobalTransporterFactory 
 * \brief General Monte Carlo transporter for domain decomposed problems.
 */
//---------------------------------------------------------------------------//
template<class Source>
class GlobalTransporterFactory
{
  public:

    //@{
    //! Typedefs.
    typedef Source                                    source_type;
    typedef SourceTraits<Source>                      ST;
    typedef typename ST::domain_type                  Domain;
    typedef Teuchos::Comm<int>                        Comm;
    //@}

    // Constructor.
    GlobalTransporterFactory() { /* ... */ }

    // Destructor.
    virutal ~GlobalTransporterFactory() { /* ... */ }

    // Creation method.
    static Teuchos::RCP<GlobalTransporter<Source> >
    create( const Teuchos::RCP<const Comm>& comm,
            const Teuchos::RCP<Domain>& domain, 
            const Teuchos::ParameterList& plist );
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_GlobalTransporterFactory_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_GLOBALTRANSPORTERFACTORY_HPP

//---------------------------------------------------------------------------//
// end MCLS_GlobalTransporterFactory.hpp
//---------------------------------------------------------------------------//

