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
 * \file MCLS_FixedPointIterationFactory_impl.hpp
 * \author Stuart R. Slattery
 * \brief Fixed point iteration factory implementation.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_FIXEDPOINTITERATIONFACTORY_IMPL_HPP
#define MCLS_FIXEDPOINTITERATIONFACTORY_IMPL_HPP

#include "MCLS_DBC.hpp"
#include "MCLS_RichardsonIteration.hpp"
#include "MCLS_SteepestDescentIteration.hpp"
#include "MCLS_MinimalResidualIteration.hpp"
#include "MCLS_RNSDIteration.hpp"

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Vector, class Matrix>
FixedPointIterationFactory<Vector,Matrix>::FixedPointIterationFactory()
{
    // Create the sovler name-to-enum map.
    d_name_map["Richardson"] = RICHARDSON;
    d_name_map["Steepest Descent"] = STEEPEST_DESCENT;
    d_name_map["MINRES"] = MINRES;
    d_name_map["RNSD"] = RNSD;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Creation method.
 */
template<class Vector, class Matrix>
Teuchos::RCP<typename FixedPointIterationFactory<Vector,Matrix>::Iteration> 
FixedPointIterationFactory<Vector,Matrix>::create( 
    const std::string& iteration_name,
    const Teuchos::RCP<Teuchos::ParameterList>& iteration_parameters )
{
    MCLS_REQUIRE( !iteration_parameters.is_null() );

    Teuchos::RCP<Iteration> iteration;

    MapType::const_iterator id = d_name_map.find( iteration_name );
    MCLS_INSIST( id != d_name_map.end(), "Iteration type not supported!" );

    switch( id->second )
    {
	case RICHARDSON:

	    iteration = Teuchos::rcp( new RichardsonIteration<Vector,Matrix>(
				       iteration_parameters ) );
	    break;

	case STEEPEST_DESCENT:

	    iteration = Teuchos::rcp( 
                new SteepestDescentIteration<Vector,Matrix>() );
	    break;

	case MINRES:

	    iteration = Teuchos::rcp( 
                new MinimalResidualIteration<Vector,Matrix>() );
	    break;

	case RNSD:

	    iteration = Teuchos::rcp( 
                new RNSDIteration<Vector,Matrix>() );
	    break;

	default:

	    throw Assertion("Iteration type not supported!");
	    break;
    }

    MCLS_ENSURE( !iteration.is_null() );

    return iteration;
}

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_FIXEDPOINTITERATIONFACTORY_IMPL_HPP

//---------------------------------------------------------------------------//
// end MCLS_FixedPointIterationFactory_impl.hpp
// ---------------------------------------------------------------------------//

