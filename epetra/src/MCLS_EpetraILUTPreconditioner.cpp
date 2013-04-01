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
 * \file MCLS_EpetraILUTPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief ILUT preconditioning for Epetra.
 */
//---------------------------------------------------------------------------//

#include "MCLS_EpetraILUTPreconditioner.hpp"
#include <MCLS_DBC.hpp>

#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Epetra_Vector.h>

#include <Ifpack_ILUT.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the valid parameters for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraILUTPreconditioner::getValidParameters() const
{
    return Teuchos::parameterList();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Get the current parameters being used for this preconditioner.
 */
Teuchos::RCP<const Teuchos::ParameterList> 
EpetraILUTPreconditioner::getCurrentParameters() const
{
    // This preconditioner has no parameters.
    return Teuchos::parameterList();
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the parameters for the preconditioner. The preconditioner will
 * modify this list with default parameters that are not defined. 
 */
void EpetraILUTPreconditioner::setParameters( 
    const Teuchos::RCP<Teuchos::ParameterList>& params )
{
    // This preconditioner has no parameters.
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Set the operator with the preconditioner.
 */
void EpetraILUTPreconditioner::setOperator( 
    const Teuchos::RCP<const matrix_type>& A )
{
    MCLS_REQUIRE( Teuchos::nonnull(A) );
    d_A = A;
}

//---------------------------------------------------------------------------//
/*! 
 * \brief Build the preconditioner.
 */
void EpetraILUTPreconditioner::buildPreconditioner()
{
    MCLS_REQUIRE( Teuchos::nonnull(d_A) );
    MCLS_REQUIRE( d_A->Filled() );

    // Build the Ifpack ILUT preconditioner.
    Ifpack_ILUT ifpack( *d_A );
    ifpack.SetParameters( params );
    ifpack.Initialize();
    ifpack.Compute();

    // Invert L and U.
    computeInverseL( ifpack.L() );
    computeInverseU( ifpack.U() );
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// end MCLS_EpetraILUTPreconditioner.cpp
//---------------------------------------------------------------------------//
