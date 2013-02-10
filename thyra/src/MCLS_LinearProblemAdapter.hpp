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
 * \brief LinearProblem adapter for Thyra
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_LINEARPROBLEMADAPTER_HPP
#define MCLS_LINEARPROBLEMADAPTER_HPP

#include <MCLS_LinearProblem.hpp>

#include <Teuchos_RCP.hpp>

#include <Epetra_Vector.h>
#include <Epetra_MultiVector.h>
#include <Epetra_RowMatrix.h>

#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class LinearProblemAdapter
 * \brief Linear system container for A*x = b.
 */
template<class Vector, class MultiVector, class Matrix>
class LinearProblemAdapter
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                      vector_type;
    typedef MultiVector                                 multivector_type;
    typedef Matrix                                      matrix_type;
    //@}

    //! Constructor.
    LinearProblemAdapter( const Teuchos::RCP<const Matrix>& A,
			  const Teuchos::RCP<MultiVector>& x,
			  const Teuchos::RCP<const MultiVector>& b )
	: d_A( A )
	, d_x( x )
	, d_b( b )
    { /* ... */ }

    // Destructor.
    ~LinearProblemAdapter()
    { /* ... */ }

    //! Get a subproblem given a LHS/RHS id.
    Teuchos::RCP<LinearProblem<Vector,Matrix> > getSubProblem( const int id );

    //! Set the linear operator.
    void setOperator( const Teuchos::RCP<const Matrix>& A )
    { d_A = A; }

    //! Set the left-hand side.
    void setLHS( const Teuchos::RCP<MultiVector>& x )
    { d_x = x; }

    //! Set the right-hand side.
    void setRHS( const Teuchos::RCP<const MultiVector>& b );
    { d_b = b; }

    //! Get the linear operator.
    Teuchos::RCP<const Matrix> getOperator() const { return d_A; }

    //! Get the left-hand side.
    Teuchos::RCP<MultiVector> getLHS() const { return d_x; }

    //! Get the right-hand side.
    Teuchos::RCP<const MultiVector> getRHS() const { return d_b; }

  private:

    // Linear operator.
    Teuchos::RCP<const Matrix> d_A;

    // Left-hand side.
    Teuchos::RCP<MultiVector> d_x;

    // Right-hand side.
    Teuchos::RCP<const MultiVector> d_b;
};

//---------------------------------------------------------------------------//
/*!
 * \brief Partial specialization for Epetra_RowMatrix.
 */
template<>
Teuchos::RCP<LinearProblem<Epetra_Vector,Epetra_RowMatrix> >
LinearProblemAdapter<Epetra_Vector,Epetra_MultiVector,Epetra_RowMatrix>
{
    return Teuchos::null;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Partial specialization for Tpetra::CrsMatrix
 */
template<class Scalar, class LO, class GO>
Teuchos::RCP<LinearProblem<Tpetra::Vector<Scalar,LO,GO>,
			   Tpetra::CrsMatrix<Scalar,LO,GO> >
LinearProblemAdapter<Tpetra::Vector<Scalar,LO,GO>,
		     Tpetra::MultiVector<Scalar,LO,GO>,
		     Tpetra::CrsMatrix<Scalar,LO,GO> >
{
    return Teuchos::null;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_LINEARPROBLEMADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblemAdapter.hpp
// ---------------------------------------------------------------------------//

