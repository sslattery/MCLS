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
 * \file MCLS_LinearProblem.hpp
 * \author Stuart R. Slattery
 * \brief Linear Problem declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_LINEARPROBLEM_HPP
#define MCLS_LINEARPROBLEM_HPP

#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"

#include <Teuchos_RCP.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class LinearProblem
 * \brief Linear system container for A*x = b.
 */
template<class Vector, class Matrix>
class LinearProblem
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                      vector_type;
    typedef Matrix                                      matrix_type;
    typedef VectorTraits<Vector>                        VT;
    typedef MatrixTraits<Vector,Matrix>                 MT;
    //@}

    // Constructor.
    LinearProblem( const Teuchos::RCP<const Matrix>& A,
		   const Teuchos::RCP<Vector>& x,
		   const Teuchos::RCP<const Vector>& b );

    // Destructor.
    ~LinearProblem();

    //! Set the linear operator.
    void setOperator( const Teuchos::RCP<const Matrix>& A );

    //! Set the left-hand side.
    void setLHS( const Teuchos::RCP<Vector>& x );

    //! Set the righ-hand side.
    void setRHS( const Teuchos::RCP<const Vector>& b );

    //! Get the linear operator.
    Teuchos::RCP<const Matrix> getOperator() const { return d_A; }

    //! Get the left-hand side.
    Teuchos::RCP<Vector> getLHS() const { return d_x; }

    //! Get the right-hand side.
    Teuchos::RCP<const Vector> getRHS() const { return d_b; }

    //! Get the residual.
    Teuchos::RCP<const Vector> getResidual() const { return d_r; }

    // Apply the linear operator to a vector.
    void applyOperator( const Vector& x, Vector& y );

    // Update the residual.
    void updateResidual();

    //! Get the status of the linear problem.
    bool status() const { return d_status; }

    //! Set the linear problem status to true.
    void setProblem() { d_status = true; }

  private:

    // Linear operator.
    Teuchos::RCP<const Matrix> d_A;

    // Left-hand side (solution vector).
    Teuchos::RCP<Vector> d_x;

    // Right-hand side.
    Teuchos::RCP<const Vector> d_b;

    // Residual r = b - A*x.
    Teuchos::RCP<Vector> d_r;

    // Boolean for linear system status. True if we are ready to solve.
    bool d_status;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_LinearProblem_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_LINEARPROBLEM_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblem.hpp
// ---------------------------------------------------------------------------//

