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

#include "MCLS_GlobalTransporter.hpp"
#include "MCLS_SourceTraits.hpp"
#include "MCLS_DomainTraits.hpp"
#include "MCLS_DomainTransporter.hpp"
#include "MCLS_DomainCommunicator.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ArrayRCP.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class SourceTransporter 
 * \brief Monte Carlo transporter for domain decomposed problems with
 * domain-to-domain commication.
 *
 * This transporter will transport the histories provided by the source and
 * all subsequent histories through the global domain until completion. All
 * communication operations occur within a set. Multiple set problems will
 * create multiple instances of this class.
 */
//---------------------------------------------------------------------------//
template<class Source>
class SourceTransporter : public GlobalTransporter<Source>
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
    typedef typename DT::bank_type                    BankType;
    typedef DomainTransporter<Domain>                 DomainTransporterType;
    typedef DomainCommunicator<Domain>                DomainCommunicatorType;
    typedef Teuchos::Comm<int>                        Comm;
    typedef Teuchos::CommRequest<int>                 Request;
    //@}

    // Constructor.
    SourceTransporter( const Teuchos::RCP<const Comm>& comm,
		       const Teuchos::RCP<Domain>& domain, 
		       const Teuchos::ParameterList& plist );

    // Assign the source.
    void assignSource( const Teuchos::RCP<Source>& source );

    // Transport the source histories and all subsequent histories through the
    // domain to completion.
    void transport();

    // Reset the state of the transporter.
    void reset();

  private:

    // Transport a source history.
    void transportSourceHistory( BankType& bank );

    // Transport a bank history.
    void transportBankHistory( BankType& bank );

    // Transport a history through the local domain.
    template<class T>
    void localHistoryTransport( T&& history );

    // Process incoming messages.
    void processMessages( BankType& bank );

    // Post communications in the binary tree.
    void postTreeCount();

    // Complete outstanding communications in the binary tree at the end of a
    // cycle.
    void completeTreeCount();

    // Update the binary tree count of completed histories.
    void updateTreeCount();

    // Send the global finished message to the children.
    void sendCompleteToChildren();

    // Control the termination of a stage.
    void controlTermination();

  private:

    // Master proc enumeration for implementation clarity.
    enum MasterIndicator { MASTER = 0 };

  private:

    // Parallel communicator for this set.
    Teuchos::RCP<const Comm> d_comm;

    // Parent process.
    int d_parent;

    // Child processes.
    std::pair<int,int> d_children;

    // Local domain.
    Teuchos::RCP<Domain> d_domain;

    // Domain transporter.
    DomainTransporterType d_domain_transporter;

    // Domain communicator.
    DomainCommunicatorType d_domain_communicator;

    // Source.
    Teuchos::RCP<Source> d_source;

    // Master-worker asynchronous communication request handles for number of
    // histories complete.
    std::pair<Teuchos::RCP<Request>,Teuchos::RCP<Request> > d_num_done_handles;

    // Master-worker reports for number of histories complete communications. 
    std::pair<Teuchos::ArrayRCP<int>,Teuchos::ArrayRCP<int> > d_num_done_report;

    // Request handle for completed work on worker nodes.
    Teuchos::RCP<Request> d_complete_handle;

    // Completion report.
    Teuchos::ArrayRCP<int> d_complete_report;

    // Total number of source histories in set.
    int d_nh;

    // Total number of histories completed in set.
    Teuchos::ArrayRCP<int> d_num_done;
    
    // Number of histories complete in the local domain.
    int d_num_run;

    // Boolean-as-integer from completion of transport calculation.
    Teuchos::ArrayRCP<int> d_complete;

    // Check frequency for history buffer communication.
    int d_check_freq;

    // Completed histories tag.
    int d_num_done_tag;

    // Completion status tag.
    int d_completion_status_tag;
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

