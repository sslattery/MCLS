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
 * \file MCLS_GlobalTransporterFactory_impl.hpp
 * \author Stuart R. Slattery
 * \brief GlobalTransporterFactory class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_GLOBALTRANSPORTERFACTORY_IMPL_HPP
#define MCLS_GLOBALTRANSPORTERFACTORY_IMPL_HPP

#include <string>

#include "MCLS_SourceTransporter.hpp"
#include "MCLS_SubdomainTransporter.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Creation method.
 */
//---------------------------------------------------------------------------//
template<class Source>
Teuchos::RCP<GlobalTransporter<Source> >
GlobalTransporterFactory<Source>::create( const Teuchos::RCP<const Comm>& comm,
        const Teuchos::RCP<Domain>& domain, 
        const Teuchos::ParameterList& plist )
{
    Teuchos::RCP<GlobalTransporter<Source> > transporter;

    if ( "Global" == plist->get<std::string>("Transport Type") )
    {
        transporter = Teuchos::rcp( 
            new SourceTransporter<Source>(comm, domain, plist) );
    }
    else if ( "Subdomain" == plist->get<std::string>("Transport Type") )
    {
        transporter = Teuchos::rcp( 
            new SubdomainTransporter<Source>(comm, domain, plist) );
    }

    return transporter;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_GLOBALTRANSPORTERFACTORY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_GlobalTransporterFactory_impl.hpp
//---------------------------------------------------------------------------//

