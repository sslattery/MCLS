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

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_OrdinalTraits.hpp>

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
// end tstSamplingTools.cpp
//---------------------------------------------------------------------------//
