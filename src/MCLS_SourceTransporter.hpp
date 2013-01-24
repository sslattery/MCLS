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
 * \file MCLS_SourceTransporter.hpp
 * \author Stuart R. Slattery
 * \brief SourceTransporter class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOURCETRANSPORTER_HPP
#define MCLS_SOURCETRANSPORTER_HPP

#include "MCLS_DomainTransporter.hpp"
#include "MCLS_DomainCommunicator.hpp"
#include "MCLS_Source.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class SourceTransporter 
 * \brief General Monte Carlo transporter for domain decomposed problems.
 *
 * This transporter will transport the histories provided by the source and
 * all subsequent histories through the global domain until completion. All
 * communication operations occur within a set. This class is based on that
 * developed by Tom Evans.
 */
//---------------------------------------------------------------------------//
template<class Domain>
class SourceTransporter
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                    domain_type;
    typedef typename Domain::HistoryType              HistoryType;
    typedef typename Domain::TallyType                TallyType;
    typedef typename Domain::BankType                 BankType;
    typedef DomainTransporter<Domain>                 DomainTransporterType;
    typedef DomainCommunicator<Domain>                DomainCommunicatorType;
    typedef Source<Domain>                            SourceType;
    typedef Teuchos::Comm<int>                        Comm;
    typedef Teuchos::CommRequest<int>                 Request;
    //@}

    // Constructor.
    SourceTransporter( const Teuchos::RCP<const Comm>& comm,
		       const Teuchos::RCP<Domain>& domain, 
		       const Teuchos::ParameterList& plist );

    // Destructor.
    ~SourceTransporter() { /* ... */ }

    // Assign the source.
    void assignSource( const Teuchos::RCP<SourceType>& source );

    // Transport the source histories and all subsequent histories through the
    // domain to completion.
    void transport();

  private:

    // Transport a source history.
    void transportSourceHistory( BankType& bank );

    // Transport a bank history.
    void transportBankHistory( BankType& bank );

    // Transport a history through the local domain.
    void localHistoryTransport( const Teuchos::RCP<HistoryType>& history, 
				BankType& bank );

    // Post communications with the set master proc for end of cycle.
    void postMasterCount();

    // Complete communications with the set master proc for end of cycle.
    void completeMasterCount();

    // Update the master count of completed histories.
    void updateMasterCount();

  private:

    // Master proc enumeration for implementation clarity.
    enum MasterIndicator { MASTER = 0 };

  private:

    // Parallel communicator for this set.
    Teuchos::RCP<const Comm> d_comm;

    // Local domain.
    Teuchos::RCP<Domain> d_domain;

    // Tally.
    Teuchos::RCP<TallyType> d_tally;

    // Domain transporter.
    DomainTransporterType d_domain_transporter;

    // Domain communicator.
    DomainCommunicatorType d_domain_communicator;

    // Source.
    Teuchos::RCP<SourceType> d_source;

    // Master-worker asynchornous communication request handles for number of
    // histories complete.
    Teuchos::Array<Teuchos::RCP<Request> > d_num_done_handles;

    // Master-worker reports for number of histories complete communications. 
    Teuchos::Array<Teuchos::RCP<int> > d_num_done_report;

    // Request handle for completed work on worker nodes.
    Teuchos::RCP<Request> > d_complete_handle;

    // Completion report.
    Teuchos::RCP<int> d_complete_report;

    // Total number of source histories in set.
    int d_nh;

    // Total number of histories completed in set.
    int d_num_done;
    
    // Total number of histories completed locally.
    int d_num_done_local;

    // Total number of histories completed from source.
    int d_num_src;

    // Number of histories complete in the local domain.
    int d_num_run;

    // Boolean-as-integer from completion of transport calculation.
    Teuchos::RCP<int> d_complete;

    // Check frequency for history buffer communication.
    int d_check_freq;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_SourceTransporter_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_SOURCETRANSPORTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_SourceTransporter.hpp
//---------------------------------------------------------------------------//

