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
 * \file MCLS_SourceTransporter_impl.hpp
 * \author Stuart R. Slattery
 * \brief SourceTransporter class implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOURCETRANSPORTER_IMPL_HPP
#define MCLS_SOURCETRANSPORTER_IMPL_HPP

#include "MCLS_DBC.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 * /
template<class Domain>
SourceTransporter<Domain>::SourceTransporter( 
    const Teuchos::RCP<const Comm>& comm,
    const Teuchos::RCP<Domain>& domain, 
    const Teuchos::ParameterList& plist )
{

}

//---------------------------------------------------------------------------//
/*!
* \brief Assign the source.
*/
template<class Domain>
void SourceTransporter<Domain>::assignSource(
    const Teuchos::RCP<SourceType>& source )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport the source histories and all subsequent histories through
 * the domain to completion.
 */
template<class Domain>
void SourceTransporter<Domain>::transport()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a source history.
 */
template<class Domain>
void SourceTransporter<Domain>::transportSourceHistory( BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a bank history.
 */
template<class Domain>
void SourceTransporter<Domain>::transportBankHistory( BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Transport a history through the local domain.
 */
template<class Domain>
void SourceTransporter<Domain>::localHistoryTransport( 
    const Teuchos::RCP<HistoryType>& history, 
    BankType& bank )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Post communications with the set master proc for end of cycle.
 */
template<class Domain>
void SourceTransporter<Domain>::postMasterCount()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Complete communications with the set master proc for end of cycle.
 */
template<class Domain>
void SourceTransporter<Domain>::completeMasterCount()
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Update the master count of completed histories.
 */
template<class Domain>
void SourceTransporter<Domain>::updateMasterCount()
{

}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SOURCETRANSPORTER_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SourceTransporter_impl.hpp
//---------------------------------------------------------------------------//

