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
 * \file MCLS_Adjoint Solver.hpp
 * \author Stuart R. Slattery
 * \brief Adjoint Monte Carlo solver declaration.l
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTSOLVER_HPP
#define MCLS_ADJOINTSOLVER_HPP

#include "MCLS_Solver.hpp"
#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_AdjointDomain.hpp"
#include "MCLS_SourceTransporter.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class Adjointsolver
 * \brief Linear solver base class.
 */
template<class Vector, class Matrix>
class AdjointSolver : public Solver
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                      vector_type;
    typedef Matrix                                      matrix_type;
    typedef VectorTraits<Vector>                        VT;
    typedef MatrixTraits<Vector,Matrix>                 MT;
    typedef LinearProblem<Vector,Matrix>                LinearProblemType;
    typedef AdjointDomain<Vector,Matrix>                DomainType;
    typedef SourceTransporter<DomainType>               TransporterType;
    typedef typename TransporterType::SourceType        SourceType;
    typedef Teuchos::Comm<int>                          Comm;
    //@}

    // Constructor.
    AdjointSolver( const Teuchos::RCP<LinearProblemType>& linear_problem,
		   const Teuchos::RCP<const Comm>& global_comm,
		   Teuchos::ParameterList& plist );

    //! Destructor.
    ~AdjointSolver { /* ... */ }

    // Solve the linear problem.
    void solve();

    // Return whether the solution has converged.
    bool isConverged();

  private:

    // Linear problem.
    Teuchos::RCP<LinearProblemType> d_linear_problem;

    // Global problem communicator.
    Teuchos::RCP<const Comm> d_global_comm;

    // Local domain.
    Teuchos::RCP<DomainType> d_domain;

    // Source transporter.
    Teuchos::RCP<TransporterType> d_transporter;

    // Source.
    Teuchos::RCP<SourceType> d_source;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AdjointSolver_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTSOLVER_HPP

//---------------------------------------------------------------------------//
// end MCLS_Adjointsolver.hpp
// ---------------------------------------------------------------------------//

