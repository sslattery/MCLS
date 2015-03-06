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
 * \file MCLS_Xorshift.hpp
 * \author Stuart R. Slattery
 * \brief 64-bit Xorshift random number generator.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_XORSHIFT_HPP
#define MCLS_XORSHIFT_HPP

#include <random>
#include <limits>

#include "MCLS_RNGTraits.hpp"

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class Xorshift
 * \brief 64-bit Xorshift random number generator.
 */
//---------------------------------------------------------------------------//
template<class uint_type = uint_fast64_t, 
	 uint_type a = 13,
	 uint_type b = 7,
	 uint_type c = 17>
class Xorshift
{
  public:

    //@{
    //! Typedefs.
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    typedef uint_type result_type;
    //@}

    //! Constructor.
    explicit Xorshift( const result_type seed )
	: d_x( seed )
    { /* ... */ }

    //! Minimum value.
    result_type min () const
    { return std::numeric_limits<result_type>::min(); }

    //! Maximum value.
    result_type max () const
    { return std::numeric_limits<result_type>::max(); }

    //! Seed the engine.
    void seed( const result_type seed )
    { d_x = seed; }

    // Get a random number.
    inline result_type operator()();

  private:

    // Random number state.
    result_type d_x;
};

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Get a random number.
 */
template<class uint_type, uint_type a, uint_type b, uint_type c>
inline uint_type Xorshift<uint_type,a,b,c>::operator()()
{
    d_x ^= d_x << a;
    d_x ^= d_x >> b;
    d_x ^= d_x << c;
    return d_x;
}

//---------------------------------------------------------------------------//
// Specialization for RNGTraits.
//---------------------------------------------------------------------------//
template<class uint_type, uint_type a, uint_type b, uint_type c>
class RNGTraits<Xorshift<uint_type,a,b,c> >
{
  public:

    //@{
    //! Typedefs.
    typedef Xorshift<uint_type,a,b,c> rng_type;
    typedef typename rng_type::uniform_int_distribution_type uniform_int_distribution_type;
    typedef typename rng_type::uniform_real_distribution_type uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<rng_type> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type
    random( rng_type& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_XORSHIFT_HPP

//---------------------------------------------------------------------------//
// end MCLS_Xorshift.hpp
//---------------------------------------------------------------------------//

