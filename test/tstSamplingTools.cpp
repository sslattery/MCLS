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
 * \file   tstSamplingTools.cpp
 * \author Stuart Slattery
 * \brief  SamplingTools class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <MCLS_config.hpp>
#include <MCLS_SamplingTools.hpp>
#include <MCLS_PRNG.hpp>
#include <MCLS_RNGTraits.hpp>
#include <MCLS_Xorshift.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_Time.hpp>

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SamplingTools, multi_bin )
{
    Teuchos::Array<double> cdf( 5, 0.0 );
    cdf[0] = 0.13;
    cdf[1] = 0.27;
    cdf[2] = 0.44;
    cdf[3] = 0.69;
    cdf[4] = 1.00;

    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.13 ) );
    TEST_EQUALITY( 1, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.27 ) );
    TEST_EQUALITY( 2, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.44 ) );
    TEST_EQUALITY( 3, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.69 ) );
    TEST_EQUALITY( 4, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 1.00 ) );

    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.00 ) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.12999999 ) );
    TEST_EQUALITY( 1, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.13000001 ) );
    TEST_EQUALITY( 1, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.26999999 ) );
    TEST_EQUALITY( 2, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.27000001 ) );
    TEST_EQUALITY( 2, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.43999999 ) );
    TEST_EQUALITY( 3, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.44000001 ) );
    TEST_EQUALITY( 3, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.68999999 ) );
    TEST_EQUALITY( 4, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.69000001 ) );
    TEST_EQUALITY( 4, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.99999999 ) );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SamplingTools, one_bin )
{
    Teuchos::Array<double> cdf( 1, 1.0 );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.13 ) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.27 ) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.44 ) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 0.69 ) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleDiscreteCDF( cdf.getRawPtr(), cdf.size(), 1.00 ) );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AliasTable, table )
{
    Teuchos::Array<double> cdf( 5, 0.0 );
    cdf[0] = 0.13;
    cdf[1] = 0.37;
    cdf[2] = 0.24;
    cdf[3] = 0.69;
    cdf[4] = 0.03;

    Teuchos::Array<int> indices( 5, 0 );
    indices[0] = 1;
    indices[1] = 3;
    indices[2] = 2;
    indices[3] = 0;
    indices[4] = 2;

    TEST_EQUALITY( 3, MCLS::SamplingTools::sampleAliasTable(
		       cdf.getRawPtr(), indices.getRawPtr(), cdf.size(),
		       3, 0.43) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::sampleAliasTable(
		       cdf.getRawPtr(), indices.getRawPtr(), cdf.size(),
		       3, 0.73) );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AliasTable, time_test )
{
    typedef MCLS::Xorshift<> RNG;
    typedef MCLS::RNGTraits<RNG> RNGT;
    typedef RNGT::uniform_int_distribution_type IntDist;
    typedef MCLS::RandomDistributionTraits<IntDist> IntTraits;
    typedef RNGT::uniform_real_distribution_type RealDist;
    typedef MCLS::RandomDistributionTraits<RealDist> RealTraits;

    int nrand = 10000000;
    MCLS::PRNG<RNG> prng( 0 );
    Teuchos::RCP<RealDist> real_dist = RealTraits::create( 0.0, 1.0 );
    Teuchos::RCP<IntDist> int_dist;
    double rand_real = 0.0;
    double rand_int = 0;
    std::cout << std::endl;

    // N = 10.
    int psize = 10;
    Teuchos::Array<double> prob( psize );
    Teuchos::Array<int> table( psize );
    for ( int i = 0; i < psize; ++i )
    {
	prob[i] = (i+1.0) / psize;
	table[i] = i;
    }

    Teuchos::Time timer("sampling");
    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	MCLS::SamplingTools::sampleDiscreteCDF( 
	    prob.getRawPtr(), psize, rand_real );
    }
    timer.stop();
    std::cout << "Discrete Sampling N = 10: " 
	      << timer.totalElapsedTime() << std::endl;

    int_dist = IntTraits::create( 0, psize-1 );
    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	rand_int = prng.random( *int_dist );
	MCLS::SamplingTools::sampleAliasTable( 
	    prob.getRawPtr(), table.getRawPtr(), psize, rand_int, rand_real );
    }
    timer.stop();
    std::cout << "Alias Sampling N = 10: " 
	      << timer.totalElapsedTime() << std::endl;
    std::cout << std::endl;

    // N = 100.
    psize = 100;
    prob.resize( psize );
    table.resize( psize );
    for ( int i = 0; i < psize; ++i )
    {
	prob[i] = (i+1.0) / psize;
	table[i] = i;
    }

    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	MCLS::SamplingTools::sampleDiscreteCDF( 
	    prob.getRawPtr(), psize, rand_real );
    }
    timer.stop();
    std::cout << "Discrete Sampling N = 100: " 
	      << timer.totalElapsedTime() << std::endl;

    int_dist = IntTraits::create( 0, psize-1 );
    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	rand_int = prng.random( *int_dist );
	MCLS::SamplingTools::sampleAliasTable( 
	    prob.getRawPtr(), table.getRawPtr(), psize, rand_int, rand_real );
    }
    timer.stop();
    std::cout << "Alias Sampling N = 100: " 
	      << timer.totalElapsedTime() << std::endl;
    std::cout << std::endl;

    // N = 1,000.
    psize = 1000;
    prob.resize( psize );
    table.resize( psize );
    for ( int i = 0; i < psize; ++i )
    {
	prob[i] = (i+1.0) / psize;
	table[i] = i;
    }

    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	MCLS::SamplingTools::sampleDiscreteCDF( 
	    prob.getRawPtr(), psize, rand_real );
    }
    timer.stop();
    std::cout << "Discrete Sampling N = 1,000: " 
	      << timer.totalElapsedTime() << std::endl;

    int_dist = IntTraits::create( 0, psize-1 );
    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	rand_int = prng.random( *int_dist );
	MCLS::SamplingTools::sampleAliasTable( 
	    prob.getRawPtr(), table.getRawPtr(), psize, rand_int, rand_real );
    }
    timer.stop();
    std::cout << "Alias Sampling N = 1,000: " 
	      << timer.totalElapsedTime() << std::endl;
    std::cout << std::endl;

    // N = 10,000.
    psize = 10000;
    prob.resize( psize );
    table.resize( psize );
    for ( int i = 0; i < psize; ++i )
    {
	prob[i] = (i+1.0) / psize;
	table[i] = i;
    }

    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	MCLS::SamplingTools::sampleDiscreteCDF( 
	    prob.getRawPtr(), psize, rand_real );
    }
    timer.stop();
    std::cout << "Discrete Sampling N = 10,000: " 
	      << timer.totalElapsedTime() << std::endl;

    int_dist = IntTraits::create( 0, psize-1 );
    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	rand_int = prng.random( *int_dist );
	MCLS::SamplingTools::sampleAliasTable( 
	    prob.getRawPtr(), table.getRawPtr(), psize, rand_int, rand_real );
    }
    timer.stop();
    std::cout << "Alias Sampling N = 10,000: " 
	      << timer.totalElapsedTime() << std::endl;
    std::cout << std::endl;

    // N = 100,000.
    psize = 100000;
    prob.resize( psize );
    table.resize( psize );
    for ( int i = 0; i < psize; ++i )
    {
	prob[i] = (i+1.0) / psize;
	table[i] = i;
    }

    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	MCLS::SamplingTools::sampleDiscreteCDF( 
	    prob.getRawPtr(), psize, rand_real );
    }
    timer.stop();
    std::cout << "Discrete Sampling N = 100,000: " 
	      << timer.totalElapsedTime() << std::endl;

    int_dist = IntTraits::create( 0, psize-1 );
    timer.start( true );
    for ( int n = 0; n < nrand; ++n )
    {
	rand_real = prng.random( *real_dist );
	rand_int = prng.random( *int_dist );
	MCLS::SamplingTools::sampleAliasTable( 
	    prob.getRawPtr(), table.getRawPtr(), psize, rand_int, rand_real );
    }
    timer.stop();
    std::cout << "Alias Sampling N = 100,000: " 
	      << timer.totalElapsedTime() << std::endl;
    std::cout << std::endl;
}

//---------------------------------------------------------------------------//
// end tstSamplingTools.cpp
//---------------------------------------------------------------------------//
