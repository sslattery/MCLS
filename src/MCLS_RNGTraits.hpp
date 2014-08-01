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
 * \file MCLS_RNGTraits.hpp
 * \author Stuart R. Slattery
 * \brief Random number generator traits.
 */
//---------------------------------------------------------------------------//

#include <random>

#include <Teuchos_RCP.hpp>

#ifndef MCLS_RNGTRAITS_HPP
#define MCLS_RNGTRAITS_HPP

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class UndefinedDistributionTraits
 * \brief Class for undefined random number distribution traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class Distribution>
struct UndefinedDistributionTraits
{
    static inline void notDefined()
    {
	return Distribution::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class DistributionTraits
 * \brief Random number distribution traits.
 */
//---------------------------------------------------------------------------//
template<class RandomDistribution>
class RandomDistributionTraits
{
  public:

    //@{
    //! Typedefs.
    typedef RandomDistribution distribution_type;
    typedef RandomDistribution::result_type result_type;
    //@}

    //! Create a random number distribution from upper and lower bounds.
    static Teuchos::RCP<RNG> create( const result_type lower_bound,
				     const result_type upper_bound )
    {
	UndefinedRNGTraits<RNG>::notDefined();
	return Teuchos::null();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedRNGTraits
 * \brief Class for undefined random number generator traits. 
 *
 * Will throw a compile-time error if these traits are not specialized.
 */
template<class RNG>
struct UndefinedRNGTraits
{
    static inline void notDefined()
    {
	return RNG::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class RNGTraits
 * \brief Random number generator traits.
 */
//---------------------------------------------------------------------------//
template<class RNG>
class RNGTraits
{
  public:

    //@{
    //! Typedefs.
    typedef RNG rng_type;
    typedef RNG::uniform_int_distribution_type uniform_int_distribution_type;
    typedef RNG::uniform_real_distribution_type uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	UndefinedRNGTraits<RNG>::notDefined();
	return Teuchos::null();
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	UndefinedRNGTraits<RNG>::notDefined();
	return 0;
    }
};

//---------------------------------------------------------------------------//
// C++11 Specializations for DistributionTraits.
//---------------------------------------------------------------------------//
template<>
class RandomDistributionTraits<std::uniform_real_distribution<double> >
{
  public:

    //@{
    //! Typedefs.
    typedef std::uniform_real_distribution<double> distribution_type;
    typedef std::uniform_real_distribution<double>::result_type result_type;
    //@}

    //! Create a random number distribution from upper and lower bounds.
    static Teuchos::RCP<RNG> create( const result_type lower_bound,
				     const result_type upper_bound )
    {
	return Teuchos::rcp( 
	    new std::uniform_real_distribution<double>(lower_bound,upper_bound) );
    }
};

//---------------------------------------------------------------------------//
template<>
class RandomDistributionTraits<std::uniform_int_distribution<int> >
{
  public:

    //@{
    //! Typedefs.
    typedef std::uniform_int_distribution<int> distribution_type;
    typedef std::uniform_int_distribution<int>::result_type result_type;
    //@}

    //! Create a random number distribution from upper and lower bounds.
    static Teuchos::RCP<RNG> create( const result_type lower_bound,
				     const result_type upper_bound )
    {
	return Teuchos::rcp( 
	    new std::uniform_int_distribution<int>(lower_bound,upper_bound) );
    }
};

//---------------------------------------------------------------------------//
// C++11 Specializations for RNGTraits.
//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::default_random_engine>
{
  public:

    //@{
    //! Typedefs.
    typedef std::default_random_engine rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::minstd_rand>
{
  public:

    //@{
    //! Typedefs.
    typedef std::minstd_rand rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::minstd_rand0>
{
  public:

    //@{
    //! Typedefs.
    typedef std::minstd_rand0 rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::mt19937>
{
  public:

    //@{
    //! Typedefs.
    typedef std::mt19937 rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::mt19937_64>
{
  public:

    //@{
    //! Typedefs.
    typedef std::mt19937_64 rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::ranlux24_base>
{
  public:

    //@{
    //! Typedefs.
    typedef std::ranlux24_base rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::ranlux48_base>
{
  public:

    //@{
    //! Typedefs.
    typedef std::ranlux48_base rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::ranlux24>
{
  public:

    //@{
    //! Typedefs.
    typedef std::ranlux24 rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::ranlux48>
{
  public:

    //@{
    //! Typedefs.
    typedef std::ranlux48 rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};


//---------------------------------------------------------------------------//
template<>
class RNGTraits<std::knuth_b>
{
  public:

    //@{
    //! Typedefs.
    typedef std::knuth_b rng_type;
    typedef std::uniform_int_distribution<int> uniform_int_distribution_type;
    typedef std::uniform_real_distribution<double> uniform_real_distribution_type;
    //@}

    //! Create a random number generator from a seed.
    static Teuchos::RCP<RNG> create( const std::size_t seed )
    {
	return Teuchos::rcp( new rng_type(seed) );
    }

    //! Get a random number from a specified distribution.
    template<class RandomDistribution>
    static inline typename RandomDistributionTraits<RandomDistribution>::result_type 
    random( RNG& rng, RandomDistribution& distribution )
    {
	return distribution( rng );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_RNGTRAITS_HPP

//---------------------------------------------------------------------------//
// end MCLS_RNGTraits.hpp
//---------------------------------------------------------------------------//

