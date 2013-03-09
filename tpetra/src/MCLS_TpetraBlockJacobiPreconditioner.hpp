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
 * \file MCLS_TpetraBlockJacobiPreconditioner.hpp
 * \author Stuart R. Slattery
 * \brief Block Jacobi preconditioning for Tpetra.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRABLOCKJACOBI_HPP
#define MCLS_TPETRABLOCKJACOBI_HPP

#include <MCLS_DBC.hpp>
#include <MCLS_Preconditioner.hpp>

#include <Teuchos_RCP.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class TpetraBlockJacobiPreconditioner
 * \brief Block-Jacobi preconditioner for Tpetra::CrsMatrix
 */
template<class Scalar, class LO, class GO>
class TpetraBlockJacobiPreconditioner
    : public Preconditioner<Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef Tpetra::Vector<Scalar,LO,GO>            vector_type;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO>         matrix_type;
    //@}

    /*!
     * \brief Constructor.
     */
    TpetraBlockJacobiPreconditioner() { /* ... */ }

    /*!
     * \brief Destructor.
     */
    ~TpetraBlockJacobiPreconditioner() { /* ... */ }

    /*!
     * \brief Set the operator with the preconditioner.
     */
    void setOperator( const Teuchos::RCP<const matrix_type>& A )
    {
	MCLS_REQUIRE( Teuchos::nonnull(A) );
	d_A = A;
    }

    // Build the preconditioner.
    void buildPreconditioner();

    /*!
     * \brief Get the preconditioner.
     */
    Teuchos::RCP<const matrix_type> getPreconditioner() const
    { return d_preconditioner; }

  private:

    // Original operator.
    Teuchos::RCP<const matrix_type> d_A;

    // Preconditioner (M^-1)
    Teuchos::RCP<matrix_type> d_preconditioner;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_TpetraBlockJacobiPreconditioner_impl.hpp"

//---------------------------------------------------------------------------//


#endif // end MCLS_TPETRABLOCKJACOBI_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraBlockJacobiPreconditioner.hpp
//---------------------------------------------------------------------------//
