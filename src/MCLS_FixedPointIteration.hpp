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
 * \file MCLS_FixedPointIteration.hpp
 * \author Stuart R. Slattery
 * \brief Fixed point ieration base class.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FIXEDPOINTITERATION_HPP
#define MCLS_FIXEDPOINTITERATION_HPP

#include <string>

#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Describable.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class FixedPointIteration
 * \brief Linear solver base class.
 */
template<class Vector, class Matrix>
class FixedPointIteration : public virtual Teuchos::Describable
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                  vector_type;
    typedef VectorTraits<Vector>                    VT;
    typedef typename VT::scalar_type                Scalar;
    typedef Matrix                                  matrix_type;
    //@}

    //! Constructor.
    FixedPointIteration() { /* ... */ }

    //! Destructor.
    virtual ~FixedPointIteration() { /* ... */ }

    //! Get the linear problem being solved by the iteration.
    virtual const LinearProblem<Vector,Matrix>& getProblem() const = 0;

    //! Get the valid parameters for this iteration.
    virtual Teuchos::RCP<const Teuchos::ParameterList> 
    getValidParameters() const = 0;

    //! Get the current parameters being used for this iteration.
    virtual Teuchos::RCP<const Teuchos::ParameterList> 
    getCurrentParameters() const = 0;

    //! Set the linear problem with the iteration.
    virtual void setProblem( 
	const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem ) = 0;

    //! Set the parameters for the iteration. The iteration will modify this
    //! list with default parameters that are not defined.
    virtual void setParameters( 
	const Teuchos::RCP<Teuchos::ParameterList>& params ) = 0;

    //! Do a single fixed point iteration.
    virtual void doOneIteration() = 0;

    //! Get the name of the fixed point iteration.
    virtual std::string name() const = 0;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_FIXEDPOINTITERATION_HPP

//---------------------------------------------------------------------------//
// end MCLS_FixedPointIteration.hpp
//---------------------------------------------------------------------------//

