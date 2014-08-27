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
 * \file MCLS_SamplingTools.hpp
 * \author Stuart R. Slattery
 * \brief SamplingTools definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SAMPLINGTOOLS_HPP
#define MCLS_SAMPLINGTOOLS_HPP

#include <algorithm>
#include <cmath>

#include "MCLS_DBC.hpp"

#include <Teuchos_ArrayView.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class SamplingTools
 * \brief Tools for sampling distributions.
 */
class SamplingTools
{
  public:

    /*
     * \brief Given a discrete CDF and random number, sample it to get the
     * output state. 
     */
    template<class T>
    static inline int
    sampleDiscreteCDF( const T* cdf,
		       const int size,
		       const T& random )
    {
	MCLS_REQUIRE( size > 0 );
	MCLS_REQUIRE( std::abs( cdf[size-1] - 1.0 ) < 1.0e-6 );
	MCLS_REQUIRE( random >= 0.0 && random <= 1.0 );
	MCLS_REMEMBER( const T *bin = std::lower_bound(cdf, cdf+size, random) );
	MCLS_ENSURE( bin - cdf >= 0 && bin - cdf < size );
	
	return std::lower_bound( cdf, cdf+size, random ) - cdf;
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SAMPLINGTOOLS_HPP

//---------------------------------------------------------------------------//
// end MCLS_SamplingTools.hpp
// ---------------------------------------------------------------------------//

