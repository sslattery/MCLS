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
#include <Teuchos_Array.hpp>

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

    /*!
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

    /*!
     * \brief Given a set of discreate probabilities, construct an alias
     * table.
     */
    static inline void createAliasTable( 
	const Teuchos::ArrayView<const double>& probs,
	const Teuchos::ArrayView<double>& alias_cdfs,
	const Teuchos::ArrayView<int>& alias_indices )
    {
	MCLS_CHECK( probs.size() == alias_cdfs.size() );
	MCLS_CHECK( probs.size() == alias_indices.size() );
	
	int np = probs.size();
	Teuchos::Array<int> lo_indices;
	Teuchos::Array<int> hi_indices;
	int hi_idx = 0;
	int lo_idx = 0;

	// First sort the probabilities by those that are greater than (1/N)
	// and those that are smaller. This will tell us which ones go in the
	// bins first.
	for ( int i = 0; i < np; ++i )
	{
	    alias_cdfs[i] = np * probs[i];
	    if ( alias_cdfs[i] < 1.0 )
	    {
		lo_indices.push_back( i );
	    }
	    else
	    {
		hi_indices.push_back( i );
	    }
	}

	// Now start creating the bins. If a probability of greater than (1/N)
	// goes in the bin, figure out the leftover probability. Sort the
	// leftover probabilities based on if they are larger or smaller than
	// (1/N).
	while ( !lo_indices.empty() && !hi_indices.empty() )
	{
	    hi_idx = hi_indices.back();
	    hi_indices.pop_back();

	    lo_idx = lo_indices.back();
	    lo_indices.pop_back();

	    alias_indices[lo_idx] = hi_idx;
	    alias_cdfs[hi_idx] = 
		alias_cdfs[hi_idx] + 
		alias_cdfs[lo_idx] - 1.0;

	    if ( alias_cdfs[hi_idx] < 1.0 )
	    {
		lo_indices.push_back( hi_idx );
	    }
	    else
	    {
		hi_indices.push_back( hi_idx );
	    }
	}
    }

    /*!
     * \brief Given an alias table and a random number, sample it to get the
     * output state.
     */
    static inline int sampleAliasTable( const double* alias_cdf,
					const int* alias_states,
					const int size,
					const int& rand_int,
					const double& rand_dbl )
    {
	MCLS_REQUIRE( size > 0 );
	MCLS_REQUIRE( 0 <= rand_int && rand_int < size );
	MCLS_REQUIRE( rand_dbl >= 0.0 && rand_dbl <= 1.0 );

	return ( rand_dbl < alias_cdf[rand_int] ) 
	    ? rand_int : alias_states[rand_int];
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//

#endif // end MCLS_SAMPLINGTOOLS_HPP

//---------------------------------------------------------------------------//
// end MCLS_SamplingTools.hpp
// ---------------------------------------------------------------------------//

