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

    TEST_EQUALITY( 0, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.13 ) );
    TEST_EQUALITY( 1, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.27 ) );
    TEST_EQUALITY( 2, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.44 ) );
    TEST_EQUALITY( 3, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.69 ) );
    TEST_EQUALITY( 4, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 1.00 ) );

    TEST_EQUALITY( 0, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.00 ) );
    TEST_EQUALITY( 0, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.12999999 ) );
    TEST_EQUALITY( 1, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.13000001 ) );
    TEST_EQUALITY( 1, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.26999999 ) );
    TEST_EQUALITY( 2, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.27000001 ) );
    TEST_EQUALITY( 2, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.43999999 ) );
    TEST_EQUALITY( 3, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.44000001 ) );
    TEST_EQUALITY( 3, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.68999999 ) );
    TEST_EQUALITY( 4, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.69000001 ) );
    TEST_EQUALITY( 4, MCLS::SamplingTools::SampleDiscreteCDF( cdf(), 0.99999999 ) );
}

//---------------------------------------------------------------------------//
// end tstSamplingTools.cpp
//---------------------------------------------------------------------------//
