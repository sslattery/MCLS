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
 * \file MCLS_SolverFactory_impl.hpp
 * \author Stuart R. Slattery
 * \brief Linear solver factory implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SOLVERFACTORY_IMPL_HPP
#define MCLS_SOLVERFACTORY_IMPL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_MonteCarloSolverManager.hpp"
#include "MCLS_MCSASolverManager.hpp"
#include "MCLS_FixedPointSolverManager.hpp"
#include "MCLS_AndersonSolverManager.hpp"

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
SolverFactory<Vector,Matrix>::SolverFactory()
{
    // Create the sovler name-to-enum map.
    d_name_map["Adjoint MC"] = ADJOINT_MC;
    d_name_map["Forward MC"] = FORWARD_MC;
    d_name_map["MCSA"] = MCSA;
    d_name_map["Fixed Point"] = FIXED_POINT;
    d_name_map["Anderson"] = ANDERSON;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Creation method.
 */
template<class Vector, class Matrix>
Teuchos::RCP<typename SolverFactory<Vector,Matrix>::Solver> 
SolverFactory<Vector,Matrix>::create( 
    const std::string& solver_name,
    const Teuchos::RCP<const Comm>& global_comm,
    const Teuchos::RCP<Teuchos::ParameterList>& solver_parameters )
{
    MCLS_REQUIRE( Teuchos::nonnull(global_comm) );
    MCLS_REQUIRE( Teuchos::nonnull(solver_parameters) );

    Teuchos::RCP<Solver> solver;

    MapType::const_iterator id = d_name_map.find( solver_name );
    MCLS_INSIST( id != d_name_map.end(), "Solver type not supported!" );

    switch( id->second )
    {
	case ADJOINT_MC:

	    solver = Teuchos::rcp(
		new MonteCarloSolverManager<Vector,Matrix,AdjointTag>( 
		    solver_parameters ) );
	    break;

	case FORWARD_MC:

	    solver = Teuchos::rcp(
		new MonteCarloSolverManager<Vector,Matrix,ForwardTag>( 
		    solver_parameters ) );
	    break;

	case MCSA:

	    solver = Teuchos::rcp( new MCSASolverManager<Vector,Matrix>( 
				       solver_parameters ) );
	    break;

	case FIXED_POINT:

	    solver = Teuchos::rcp( new FixedPointSolverManager<Vector,Matrix>( 
				       solver_parameters ) );
	    break;

	case ANDERSON:

	    solver = Teuchos::rcp( new AndersonSolverManager<Vector,Matrix,>( 
				       solver_parameters ) );
	    break;

	default:

	    throw Assertion("Solver type not supported!");
	    break;
    }

    MCLS_ENSURE( Teuchos::nonnull(solver) );

    return solver;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_SOLVERFACTORY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_SolverFactory_impl.hpp
// ---------------------------------------------------------------------------//

