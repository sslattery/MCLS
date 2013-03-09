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
 * \file MCLS_Preconditioner.hpp
 * \author Stuart R. Slattery
 * \brief Linear solver manager base class.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_PRECONDITIONER_HPP
#define MCLS_PRECONDITIONER_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ParameterList.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class Preconditioner
 * \brief Preconditioner base class.
 */
template<class Matrix>
class Preconditioner : public virtual Teuchos::Describable
{
  public:

    //@{
    //! Typedefs.
    typedef Matrix                                  matrix_type;
    //@}

    //! Constructor.
    Preconditioner() { /* ... */ }

    //! Destructor.
    virtual ~Preconditioner() { /* ... */ }

    //! Get the valid parameters for this preconditioner.
    virtual Teuchos::RCP<const Teuchos::ParameterList> 
    getValidParameters() const = 0;

    //! Get the current parameters being used for this preconditioner.
    virtual Teuchos::RCP<const Teuchos::ParameterList> 
    getCurrentParameters() const = 0;

    //! Set the parameters for the preconditioner. The preconditioner will
    //! modify this list with default parameters that are not defined.
    virtual void setParameters( 
	const Teuchos::RCP<Teuchos::ParameterList>& params ) = 0;

    //! Set the operator with the preconditioner.
    virtual void setOperator( const Teuchos::RCP<const Matrix>& A ) = 0;

    //! Get the operator begin preconditioned.
    virtual const Matrix& getOperator() const = 0;

    //! Get the preconditioner.
    virtual Teuchos::RCP<const Matrix> getPreconditioner() const = 0;

    //! Build the preconditioner.
    virtual void buildPreconditioner() = 0;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_PRECONDITIONER_HPP

//---------------------------------------------------------------------------//
// end MCLS_Preconditioner.hpp
//---------------------------------------------------------------------------//

