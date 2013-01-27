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
 * \file MCLS_MCSolver.hpp
 * \author Stuart R. Slattery
 * \brief Monte Carlo solver declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MCSOLVER_HPP
#define MCLS_MCSOLVER_HPP

#include "MCLS_Solver.hpp"
#include "MCLS_RNGControl.hpp"
#include "MCLS_SourceTransporter.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MCSolver
 * \brief Monte Carlo Linear Solver. 
 *
 * The domain type indicates the solver type. For example, templating this
 * class on the AdjointDomain will solve the system using the analog adjoint
 * Neumann-Ulam method.
 */
template<class Domain>
class MCSolver
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                      domain_type;
    typedef typename Domain::TallyType                  TallyType;
    typedef SourceTransporter<Domain>                   TransporterType;
    typedef typename TransporterType::SourceType        SourceType;
    typedef typename TransporterType::HistoryType       HistoryType;
    typedef Teuchos::Comm<int>                          Comm;
    //@}

    // Constructor.
    MCSolver( const Teuchos::RCP<const Comm>& set_comm,
	      const Teuchos::RCP<Teuchos::ParameterList>& plist,
	      int seed = 433494437 );

    //! Destructor.
    ~MCSolver { /* ... */ }

    // Solve the linear problem.
    void solve();

    // Set the domain.
    void setDomain( const Teuchos::RCP<Domain>& domain );

    // Set the source.
    void setSource( const Teuchos::RCP<SourceType>& source );

  private:

    // Set constant communicator.
    Teuchos::RCP<const Comm> d_set_comm;

    // Problem parameters.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Random number seed.
    int seed;

    // Random number controller.
    Teuchos::RCP<RNGControl> d_rng_control;

    // Domain.
    Teuchos::RCP<Domain> d_domain;

    // Tally.
    Teuchos::RCP<TallyType> d_tally;

    // SourceType transporter.
    Teuchos::RCP<TransporterType> d_transporter;

    // SourceType.
    Teuchos::RCP<SourceType> d_source;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_MCSolver_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MCSOLVER_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSolver.hpp
// ---------------------------------------------------------------------------//
