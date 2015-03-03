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
 * \file MCLS_SolverFactory.hpp
 * \author Stuart R. Slattery
 * \brief Linear solver factory declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOLVERFACTORY_HPP
#define MCLS_SOLVERFACTORY_HPP

#include <string>

#include "MCLS_SolverManager.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Describable.hpp>

#include <unordered_map>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class SolverFactory
 * \brief Factory class for generating solver managers.
 */
template<class Vector, class Matrix>
class SolverFactory : public virtual Teuchos::Describable
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                    vector_type;
    typedef Matrix                                    matrix_type;
    typedef SolverManager<Vector,Matrix>              Solver;     
    typedef std::unordered_map<std::string,int>       MapType;
    //@}

    //! Constructor.
    SolverFactory();

    // Creation method.
    Teuchos::RCP<Solver> 
    create( const std::string& solver_name,
	    const Teuchos::RCP<Teuchos::ParameterList>& solver_parameters );

  private:

    // Solver enum.
    enum MCLSSolverType {
	ADJOINT_MC,
        FORWARD_MC,
	ADJOINT_MCSA,
	FORWARD_MCSA,
	ADJOINT_ANDERSON,
	FORWARD_ANDERSON,
        FIXED_POINT
    };

    // String name to enum/integer map.
    MapType d_name_map;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_SolverFactory_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_SOLVERFACTORY_HPP

//---------------------------------------------------------------------------//
// end MCLS_SolverFactory.hpp
// ---------------------------------------------------------------------------//

