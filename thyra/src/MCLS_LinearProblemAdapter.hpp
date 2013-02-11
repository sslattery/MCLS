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

#include <MCLS_DBC.hpp>
#include <MCLS_LinearProblem.hpp>

#include "MCLS_MultiVectorTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Describable.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class LinearProblemAdapter
 * \brief Linear system container for A*x = b.
 *
 * This container holds a blocked linear system and provides access to the
 * individual linear systems contained within for MCLS solves.
 */
template<class MultiVector, class Matrix>
class LinearProblemAdapter : public virtual Teuchos::Describable
{
  public:

    //@{
    //! Typedefs.
    typedef MultiVector                                 multivector_type;
    typedef MultiVectorTraits<MultiVector>              MVT;
    typedef typename MVT::vector_type                   Vector;
    typedef Matrix                                      matrix_type;
    //@}

    //! Default constructor.
    LinearProblemAdapter() { /* ... */ }

    //! Constructor.
    LinearProblemAdapter( const Teuchos::RCP<const Matrix>& A,
			  const Teuchos::RCP<MultiVector>& x,
			  const Teuchos::RCP<const MultiVector>& b,
			  const int num_problems )
	: d_A( A )
	, d_x( x )
	, d_b( b )
	, d_num_problems( num_problems )
    { /* ... */ }

    //! Destructor.
    ~LinearProblemAdapter() { /* ... */ }

    //! Get a subproblem given a LHS/RHS id.
    Teuchos::RCP<LinearProblem<Vector,Matrix> > getSubProblem( const int id );

    //! Get the number of LHS/RHS in the problem.
    int getNumSubProblems() const { return d_num_problems; }  

    //! Set the number of LHS/RHS in the problem.
    void setNumSubProblems( const int num_problems ) const 
    { d_num_problems = num_problems; }  

    //! Set the linear operator.
    void setOperator( const Teuchos::RCP<const Matrix>& A ) { d_A = A; }

    //! Set the left-hand side.
    void setLHS( const Teuchos::RCP<MultiVector>& x ) { d_x = x; }

    //! Set the right-hand side.
    void setRHS( const Teuchos::RCP<const MultiVector>& b ) { d_b = b; }

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

    // Number of LHS/RHS in the problem.
    int d_num_problems;
};

//---------------------------------------------------------------------------//
/*!
 * \brief Partial specialization for Epetra_RowMatrix.
 */
template<>
Teuchos::RCP<LinearProblem<Epetra_Vector,Epetra_RowMatrix> >
LinearProblemAdapter<Epetra_Vector,
		     Epetra_MultiVector,
		     Epetra_RowMatrix>::getSubProblem( const int id )
{
    Require( id < d_x->NumVectors() );
    Require( id < d_b->NumVectors() );

    Teuchos::RCP<Epetra_Vector> vector_x = Teuchos::rcp( (*d_x)(id), false );
    Teuchos::RCP<const Epetra_Vector> vector_b = 
	Teuchos::rcp( (*d_b)(id), false );

    return Teuchos::rcp( new LinearProblem<Epetra_Vector,Epetra_RowMatrix>(
			     d_A, vector_x, vector_b) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Partial specialization for Tpetra::CrsMatrix<double,int,int>
 */
Teuchos::RCP<LinearProblem<Tpetra::Vector<double,int,int>,
			   Tpetra::CrsMatrix<double,int,int> > >
LinearProblemAdapter<Tpetra::Vector<double,int,int>,
		     Tpetra::MultiVector<double,int,int>,
		     Tpetra::CrsMatrix<double,int,int> >::getSubProblem(
			 const int id )
{
    Require( id < d_x->getNumVectors() );
    Require( id < d_b->getNumVectors() );

    Teuchos::RCP<Tpetra::Vector<double,int,int> > vector_x = 
	Teuchos::rcp( d_x->getVectorNonConst(id), false );
    Teuchos::RCP<const Tpetra::Vector<double,int,int> > vector_b = 
	Teuchos::rcp( d_b->getVector(id), false );

    return Teuchos::rcp( new LinearProblem<Tpetra::Vector<double,int,int>,
					   Tpetra::CrsMatrix<double,int,int> >(
					       d_A, vector_x, vector_b) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Partial specialization for Tpetra::CrsMatrix<double,int,long>
 */
Teuchos::RCP<LinearProblem<Tpetra::Vector<double,int,long>,
			   Tpetra::CrsMatrix<double,int,long> > >
LinearProblemAdapter<Tpetra::Vector<double,int,long>,
		     Tpetra::MultiVector<double,int,long>,
		     Tpetra::CrsMatrix<double,int,long> >::getSubProblem(
			 const int id )
{
    Require( id < d_x->getNumVectors() );
    Require( id < d_b->getNumVectors() );

    Teuchos::RCP<Tpetra::Vector<double,int,long> > vector_x = 
	Teuchos::rcp( d_x->getVectorNonConst(id), false );
    Teuchos::RCP<const Tpetra::Vector<double,int,long> > vector_b = 
	Teuchos::rcp( d_b->getVector(id), false );

    return Teuchos::rcp( new LinearProblem<Tpetra::Vector<double,int,long>,
					   Tpetra::CrsMatrix<double,int,long> >(
					       d_A, vector_x, vector_b) );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_LINEARPROBLEMADAPTER_HPP

//---------------------------------------------------------------------------//
// end MCLS_LinearProblemAdapter.hpp
// ---------------------------------------------------------------------------//

