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
 * \file MCLS_UniformSource.hpp
 * \author Stuart R. Slattery
 * \brief UniformSource class declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_UNIFORMSOURCE_HPP
#define MCLS_UNIFORMSOURCE_HPP

#include "MCLS_Source.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class UniformSource 
 * \brief Uniform sampling history source.
 *
 * This class and inheritance structure is based on that developed by Tom
 * Evans. 
 */
//---------------------------------------------------------------------------//
template<class Domain>
class UniformSource : public Source<Domain>
{
  public:

    //@{
    //! Typedefs.
    typedef Source<Domain>                               Base;
    typedef Domain                                       domain_type;
    typedef typename Domain::HistoryType                 HistoryType;
    typedef typename Domain::VectorType                  VectorType;
    typedef Teuchos::Comm<int>                           Comm;
    typedef RNGControl::RNG                              RNG;
    //@}

    // Constructor.
    UniformSource( const Teuchos::RCP<VectorType>& b,
		   const Teuchos::RCP<Domain>& domain,
		   const Teuchos::RCP<RNGControl>& rng_control,
		   const Teuchos::RCP<const Comm>& comm,
		   const Teuchos::ParameterList& plist );

    // Destructor.
    ~UniformSource() { /* ... */ }

    //! Get a history from the source.
    Teuchos::RCP<HistoryType> getHistory();

    //! Return whether the source has emitted all histories.
    bool empty() const;

    //! Get the number of source histories left in the local domain
    int numToTransport() const;

    //! Get the number of source histories left in the set.
    int numToTransportInSet() const;

  private:

    // Make a globally unique random number generator for this proc.
    void makeRNG();

  private:

    // Communicator for this set.
    Teuchos::RCP<const Comm> d_comm;

    // RNG stream offset.
    int d_rng_stream;

    // Number of requested histories.
    int d_nh_requested;

    // Number of total histories.
    int d_nh_total;
    
    // Local number of histories.
    int d_nh_domain;

    // History weight.
    double d_source_weight;

    // Number of histories left in the local domain.
    int d_nh_left;

    // Number of histories emitted in the local domain.
    int d_nh_emitted;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_UniformSource_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_UNIFORMSOURCE_HPP

//---------------------------------------------------------------------------//
// end MCLS_UniformSource.hpp
//---------------------------------------------------------------------------//

