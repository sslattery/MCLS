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
 * \file MCLS_SubdomainTransporter_impl.hpp
 * \author Stuart R. Slattery
 * \brief SubdomainTransporter class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SUBDOMAINTRANSPORTER_IMPL_HPP
#define MCLS_SUBDOMAINTRANSPORTER_IMPL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_CommTools.hpp"
#include "MCLS_Events.hpp"

#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_Ptr.hpp>
#include <Teuchos_OrdinalTraits.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Source>
SubdomainTransporter<Source>::SubdomainTransporter( 
    const Teuchos::RCP<const Comm>& comm,
    const Teuchos::RCP<Domain>& domain, 
    const Teuchos::ParameterList& plist )
    : d_comm( comm )
    , d_domain( domain )
    , d_domain_transporter( d_domain, plist )
{
    MCLS_REQUIRE( Teuchos::nonnull(d_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(d_domain) );
}

//---------------------------------------------------------------------------//
/*!
* \brief Assign the source.
*/
template<class Source>
void SubdomainTransporter<Source>::assignSource(
    const Teuchos::RCP<Source>& source,
    const double relative_weight_cutoff )
{
    MCLS_REQUIRE( Teuchos::nonnull(source) );
    d_source = source;
    d_domain_transporter.setCutoff( relative_weight_cutoff );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport the source histories through the local domain to
 * completion.
 */
template<class Source>
void SubdomainTransporter<Source>::transport()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_source) );

    // Transport all source histories through the local domain until completion.
    while (  !ST::empty(*d_source) )
    {
        // Get a history from the source.
        Teuchos::RCP<HistoryType> history = ST::getHistory( *d_source );
        MCLS_CHECK( Teuchos::nonnull(history) );
        MCLS_CHECK( HT::alive(*history) );

        // Do local transport.
        d_domain_transporter.transport( *history );
        MCLS_CHECK( !HT::alive(*history) );
        MCLS_CHECK( Event::CUTOFF == HT::event(*history) ||
                    Event::BOUNDARY == HT::event(*history) );
    }

    // Barrier before continuing.
    d_comm->barrier();

    MCLS_ENSURE( ST::empty(*d_source) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SUBDOMAINTRANSPORTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SubdomainTransporter_impl.hpp
//---------------------------------------------------------------------------//

