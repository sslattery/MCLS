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
 * \file MCLS_SubdomainTransporter.hpp
 * \author Stuart R. Slattery
 * \brief SubdomainTransporter class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SUBDOMAINTRANSPORTER_HPP
#define MCLS_SUBDOMAINTRANSPORTER_HPP

#include "MCLS_GlobalTransporter.hpp"
#include "MCLS_SourceTraits.hpp"
#include "MCLS_DomainTraits.hpp"
#include "MCLS_DomainTransporter.hpp"
#include "MCLS_DomainCommunicator.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class SubdomainTransporter 
 * \brief Monte Carlo transporter for domain decomposed problems with only
 * subdomain solutions.
 *
 * This transporter willtransport the histories provided by the source and all
 * subsequent histories through the local domain until completion. No
 * communication operations occur within a set. Multiple set problems will
 * create multiple instances of this class.
 */
//---------------------------------------------------------------------------//
template<class Source>
class SubdomainTransporter : public GlobalTransporter<Source>
{
  public:

    //@{
    //! Typedefs.
    typedef Source                                    source_type;
    typedef SourceTraits<Source>                      ST;
    typedef typename ST::domain_type                  Domain;
    typedef DomainTraits<Domain>                      DT;
    typedef typename DT::history_type                 HistoryType;
    typedef HistoryTraits<HistoryType>                HT;
    typedef DomainTransporter<Domain>                 DomainTransporterType;
    typedef Teuchos::Comm<int>                        Comm;
    //@}

    // Constructor.
    SubdomainTransporter( const Teuchos::RCP<const Comm>& comm,
                          const Teuchos::RCP<Domain>& domain, 
                          const Teuchos::ParameterList& plist );

    // Destructor.
    ~SubdomainTransporter() { /* ... */ }

    // Assign the source.
    void assignSource( const Teuchos::RCP<Source>& source, 
		       const double relative_weight_cutoff );

    // Transport the source histories and all subsequent histories through the
    // domain to completion.
    void transport();

    // Reset the state of the transporter.
    void reset();

  private:

    // Parallel communicator for this set.
    Teuchos::RCP<const Comm> d_comm;

    // Local domain.
    Teuchos::RCP<Domain> d_domain;

    // Domain transporter.
    DomainTransporterType d_domain_transporter;

    // Source.
    Teuchos::RCP<Source> d_source;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_SubdomainTransporter_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_SUBDOMAINTRANSPORTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_SubdomainTransporter.hpp
//---------------------------------------------------------------------------//

