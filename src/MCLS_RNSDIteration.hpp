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
 * \file MCLS_RNSDIteration.hpp
 * \author Stuart R. Slattery
 * \brief Residual norm steepest descent iteration declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_RNSDITERATION_HPP
#define MCLS_RNSDITERATION_HPP

#include <string>

#include "MCLS_FixedPointIteration.hpp"
#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class RNSDIteration
 * \brief Residual norm steepest descent one dimensional projection iteration
 * for non-singular problems. 
 */
template<class Vector, class Matrix>
class RNSDIteration : public FixedPointIteration<Vector,Matrix>
{
  public:

    //@{
    //! Typedefs.
    typedef FixedPointIteration<Vector,Matrix>            Base;
    typedef Vector                                        vector_type;
    typedef VectorTraits<Vector>                          VT;
    typedef typename VT::scalar_type                      Scalar;
    typedef Matrix                                        matrix_type;
    typedef LinearProblem<Vector,Matrix>                  LinearProblemType;
    //@}

    // Default constructor. setProblem() must be called before solve().
    RNSDIteration();

    // Constructor.
    RNSDIteration( const Teuchos::RCP<LinearProblemType>& problem );

    //! Destructor.
    ~RNSDIteration() { /* ... */ }

    //! Get the linear problem being solved by the manager.
    const LinearProblem<Vector,Matrix>& getProblem() const
    { return *d_problem; }

    // Get the valid parameters for this manager.
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

    //! Get the current parameters being used for this manager.
    Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters() const
    { return Teuchos::parameterList(); }

    // Set the linear problem with the manager.
    void setProblem( 
	const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem );

    // Set the parameters for the manager. The manager will modify this list
    // with default parameters that are not defined.
    void setParameters( const Teuchos::RCP<Teuchos::ParameterList>& params );

    // Do a single fixed point iteration. Must update the residual.
    void doOneIteration();

    //! Get the name of the fixed point iteration.
    std::string name() const 
    { return std::string("Residual Norm Steepest Descent"); }

  private:

    // Linear problem
    Teuchos::RCP<LinearProblemType> d_problem;

    // Search subspace
    Teuchos::RCP<Vector> d_v;

    // Constraint subspace;
    Teuchos::RCP<Vector> d_w;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_RNSDIteration_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_RNSDITERATION_HPP

//---------------------------------------------------------------------------//
// end MCLS_RNSDIteration.hpp
//---------------------------------------------------------------------------//

