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
 * \file MCLS_EpetraBlockJacobiPreconditioner.hpp
 * \author Stuart R. Slattery
 * \brief Block Jacobi preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_EPETRABLOCKJACOBI_HPP
#define MCLS_EPETRABLOCKJACOBI_HPP

#include <MCLS_DBC.hpp>
#include <MCLS_Preconditioner.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_Array.hpp>

#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class EpetraBlockJacobiPreconditioner
 * \brief Block-Jacobi preconditioner for Epetra::CrsMatrix
 */
class EpetraBlockJacobiPreconditioner : public Preconditioner<Epetra_RowMatrix>
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_RowMatrix  matrix_type;
    //@}

    // Constructor.
    EpetraBlockJacobiPreconditioner(
	const Teuchos::RCP<Teuchos::ParameterList>& params );

    //! Destructor.
    ~EpetraBlockJacobiPreconditioner() { /* ... */ }

    // Get the valid parameters for this preconditioner.
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

    // Get the current parameters being used for this preconditioner.
    Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters() const;

    // Set the parameters for the preconditioner. The preconditioner will
    // modify this list with default parameters that are not defined.
    void setParameters( const Teuchos::RCP<Teuchos::ParameterList>& params );

    // Set the operator with the preconditioner.
    void setOperator( const Teuchos::RCP<const matrix_type>& A );

    // Get the operator set with the preconditoner.
    const matrix_type& getOperator() const { return *d_A; }

    // Build the preconditioner.
    void buildPreconditioner();

    //! Get the preconditioner.
    Teuchos::RCP<const matrix_type> getPreconditioner() const
    { return d_preconditioner; }

  private:

    // Invert a Teuchos::SerialDenseMatrix block.
    void invertSerialDenseMatrix( 
	Teuchos::SerialDenseMatrix<int,double>& block );

    // Get a row of an operator in a block given a global row and block column
    // indices. 
    void getBlockRowFromGlobal( Teuchos::SerialDenseMatrix<int,double>& block,
                                const int block_row,
                                const Teuchos::RCP<const matrix_type>& matrix,
                                const int global_row,
                                const Teuchos::Array<int>& global_cols );

  private:

    // Parameter list.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Original operator.
    Teuchos::RCP<const matrix_type> d_A;

    // Preconditioner (M^-1)
    Teuchos::RCP<Epetra_CrsMatrix> d_preconditioner;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_EPETRABLOCKJACOBI_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraBlockJacobiPreconditioner.hpp
//---------------------------------------------------------------------------//
