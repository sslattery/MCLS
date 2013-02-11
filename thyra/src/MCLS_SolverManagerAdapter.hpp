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
 * \file MCLS_SolverManagerAdapter.hpp
 * \author Stuart R. Slattery
 * \brief Linear solver manager adapter class for Thyra blocked systems.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOLVERMANAGERADAPTER_HPP
#define MCLS_SOLVERMANAGERADAPTER_HPP

#include <MCLS_VectorTraits.hpp>
#include <MCLS_SolverManager.hpp>

#include "MCLS_LinearProblemAdapter.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>

#include <Thyra_SolveSupportTypes.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class SolverManagerAdapter
 * \brief SolverManager adapter for Thyra blocked systems.
 */
template<class Vector, class MultiVector, class Matrix>
class SolverManagerAdapter : public virtual Teuchos::Describable
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                      vector_type;
    typedef typename VectorTraits<Vector>::scalar_type  Scalar;
    typedef MultiVector                                 multivector_type;
    typedef Matrix                                      matrix_type;
    //@}

    // Constructor.
    SolverManagerAdapter( 
	const Teuchos::RCP<SolverManager<Vector,Matrix> >& solver );

    //! Destructor.
    ~SolverManagerAdapter() { /* ... */ }

    //! Get the blocked linear problem being solved by the manager.
    const LinearProblem<Vector,Matrix>& getProblem() const
    { return d_problem; }

    //! Get the valid parameters for this manager.
    Teuchos::RCP<const Teuchos::ParameterList> 
    getValidParameters() const
    { return d_solver->getValidParameters(); }

    //! Get the current parameters being used for this manager.
    Teuchos::RCP<const Teuchos::ParameterList> 
    getCurrentParameters() const
    { return d_solver->getCurrentParameters(); }

    // Set the linear problem with the manager.
    void setProblem( 
	const Teuchos::RCP<LinearProblemAdapter<Vector,MultiVector,Matrix> >& problem );

    //! Set the parameters for the manager. The manager will modify this list
    //! with default parameters that are not defined.
    void setParameters( 
	const Teuchos::RCP<Teuchos::ParameterList>& params )
    { d_solver->setParameters(); }

    // Solve the blocked linear problem. Return true if the solution converged
    // for all blocks. False if it did not.
    Thyra::SolveStatus<typename VectorTraits<Vector>::scalar_type> solve();

  private:

    // MCLS linear solver.
    Teuchos::RCP<SolverManager<Vector,Matrix> > d_solver;

    // Blocked linear problem.
    Teuchos::RCP<LinearProblemAdapter<Vector,MultiVector,Matrix> > d_problem;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_SolverManagerAdapter_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_SOLVERMANAGERADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_SolverManagerAdapter.hpp
//---------------------------------------------------------------------------//

