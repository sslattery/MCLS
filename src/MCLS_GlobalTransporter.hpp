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
 * \file MCLS_GlobalTransporter.hpp
 * \author Stuart R. Slattery
 * \brief GlobalTransporter class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_GLOBALTRANSPORTER_HPP
#define MCLS_GLOBALTRANSPORTER_HPP

#include <Teuchos_RCP.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class GlobalTransporter 
 * \brief General Monte Carlo transporter for domain decomposed problems.
 */
//---------------------------------------------------------------------------//
template<class Source>
class GlobalTransporter
{
  public:

    //@{
    //! Typedefs.
    typedef Source                                    source_type;
    //@}

    // Constructor.
    GlobalTransporter() { /* ... */ }

    // Destructor.
    virtual ~GlobalTransporter() { /* ... */ }

    // Assign the source.
    virtual void assignSource( const Teuchos::RCP<Source>& source, 
                               const double relative_weight_cutoff ) = 0;

    // Transport the source histories and all subsequent histories through the
    // domain to completion.
    virtual void transport() = 0;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_GLOBALTRANSPORTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_GlobalTransporter.hpp
//---------------------------------------------------------------------------//

