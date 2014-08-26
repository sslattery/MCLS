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
 * \file MCLS_TpetraParaSailsPreconditioner.hpp
 * \author Stuart R. Slattery
 * \brief ParaSails preconditioning for Tpetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAPARASAILS_HPP
#define MCLS_TPETRAPARASAILS_HPP

#include <MCLS_Preconditioner.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Time.hpp>

#include <Tpetra_CrsMatrix.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class TpetraParaSailsPreconditioner
 * \brief ParaSails sparse approximate inverse preconditioner for
 * Tpetra_RowMatrix.
 */
template<class Scalar, class LO, class GO>
class TpetraParaSailsPreconditioner : 
	public Preconditioner<Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> matrix_type;
    //@}

    // Constructor.
    TpetraParaSailsPreconditioner(
        const Teuchos::RCP<Teuchos::ParameterList>& params );

    //! Destructor.
    ~TpetraParaSailsPreconditioner() { /* ... */ }

    // Get the valid parameters for this preconditioner.
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

    // Get the current parameters being used for this preconditioner.
    Teuchos::RCP<const Teuchos::ParameterList> getCurrentParameters() const;

    // Set the parameters for the preconditioner. The preconditioner will
    // modify this list with default parameters that are not defined.
    void setParameters( const Teuchos::RCP<Teuchos::ParameterList>& params );

    // Set the operator with the preconditioner.
    void setOperator( const Teuchos::RCP<const matrix_type>& A );

    // Set the operator with the preconditioner.
    const matrix_type& getOperator() const { return *d_A; }

    // Build the preconditioner.
    void buildPreconditioner();

    //! Get the left preconditioner.
    Teuchos::RCP<const matrix_type> getLeftPreconditioner() const
    { return d_preconditioner; }

    //! Get the right preconditioner.
    Teuchos::RCP<const matrix_type> getRightPreconditioner() const
    { return d_preconditioner; }

  private:

    // Parameter list.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Original operator.
    Teuchos::RCP<const matrix_type> d_A;

    // Preconditioner (M^-1)
    Teuchos::RCP<matrix_type> d_preconditioner;

    // Preconditioner creation timer.
    Teuchos::RCP<Teuchos::Time> d_prec_timer;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_TpetraParaSailsPreconditioner_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_TPETRAPARASAILS_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraParaSailsPreconditioner.hpp
//---------------------------------------------------------------------------//
