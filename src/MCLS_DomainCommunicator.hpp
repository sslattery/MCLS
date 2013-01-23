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
 * \file MCLS_DomainCommunicator.hpp
 * \author Stuart R. Slattery
 * \brief DomainCommunicator class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_DOMAINCOMMUNICATOR_HPP
#define MCLS_DOMAINCOMMUNICATOR_HPP

#include "MCLS_HistoryBuffer.hpp"
#include "MCLS_CommHistoryBuffer.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class DomainCommunicator 
 * \brief Structure for communicating histories amongst domains in a set. 
 *
 * Tom Evans is responsible for the design of this class.
 */
//---------------------------------------------------------------------------//
template<class Domain>
class DomainCommunicator
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                       domain_type;
    typedef typename Domain::HistoryType                 HistoryType;
    typedef typename Domain::BankType                    BankType;
    typedef HistoryBuffer<HistoryType>                   HistoryBufferType;
    typedef SendHistoryBuffer<HistoryType>               SendBuffer;
    typedef ReceiveHistoryBuffer<HistoryType>            ReceiveBuffer;
    typedef Teuchos::Comm<int>                           Comm;
    //@}

    //! Communication result.
    struct Result
    {
        bool sent;
        int  destination;
    };

  public:

    // Constructor.
    DomainCommunicator( const Teuchos::RCP<Domain>& domain,
			const Teuchos::RCP<const Comm>& comm, 
			const Teuchos::ParameterList& plist );

    // Destructor.
    ~DomainCommunicator()
    { /* ... */ }

    // Buffer and send a history.
    const Result& communicate( const Teuchos::RCP<HistoryType>& history );

    // Send all buffers that are not empty.
    int send();

    // Flush all buffers whether they are empty or not.
    int flush();

    // Post receives.
    void post();

    // Wait on receive buffers.
    int wait( BankType& bank );

    // Receive buffers and repost.
    int checkAndPost( BankType& bank );

    // Status of send buffers.
    bool sendStatus();

    // Status of receive buffers.
    bool receiveStatus();

    // End communication.
    void end();

    //! Particle buffer size.
    std::size_t maxBufferSize() const
    { return HistoryBufferType::maxNum(); }

    // Number of particles in all buffers.
    std::size_t sendBufferSize() const;

    // Get a send buffer by local id.
    const SendBuffer& sendBuffer( int n ) const
    { return d_sends[n]; }

    // Get a receive buffer by local id.
    const ReceiveBuffer& receiveBuffer( int n ) const
    { return d_receives[n]; }

  private:

    // Local domain.
    Teuchos::RCP<Domain> d_domain;

    // Set-constant communicator for domain-to-domain communcation.
    Teuchos::RCP<const Comm> d_comm;

    // Send buffers.
    Teuchos::Array<SendBuffer> d_sends;

    // Receive buffers.
    Teuchos::Array<ReceiveBuffer> d_receives;

    // Number of neighbors we are sending to.
    int d_num_send_neighbors;

    // Number of neighbors we are receiving from.
    int d_num_receive_neighbors;

    // Result of a history communication.
    Result d_result;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_DomainCommunicator_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_DOMAINCOMMUNICATOR_HPP

//---------------------------------------------------------------------------//
// end MCLS_DomainCommunicator.hpp
//---------------------------------------------------------------------------//

