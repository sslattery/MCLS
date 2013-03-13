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
 * \file   tstBlockInversion.cpp
 * \author Stuart Slattery
 * \brief  LAPACK block inversion unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <sstream>
#include <stdexcept>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_OrdinalTraits.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>
#include <Teuchos_LAPACK.hpp>

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( LAPACK, block_inversion )
{
    // Build a 4x4 block.
    int m = 4;
    int n = 4;
    Teuchos::SerialDenseMatrix<int,double> block( m, n );

    block(0,0) = 3.2;
    block(0,1) = -1.43;
    block(0,2) = 2.98;
    block(0,3) = 0.32;

    block(1,0) = -4.12;
    block(1,1) = -7.53;
    block(1,2) = 1.44;
    block(1,3) = -3.72;

    block(2,0) = 4.24;
    block(2,1) = -6.42;
    block(2,2) = 1.82;
    block(2,3) = 2.67;

    block(3,0) = -0.23;
    block(3,1) = 5.8;
    block(3,2) = 1.13;
    block(3,3) = -3.73;

    // Make a LAPACK object.
    Teuchos::LAPACK<int,double> lapack;

    // Compute the LU-factorization of the block.
    Teuchos::ArrayRCP<int> ipiv( block.numRows() );
    int info = 0;
    int lda = m;
    lapack.GETRF( m, n, block.values(), lda, ipiv.getRawPtr(), &info );
    TEST_EQUALITY( info, 0 );

    // Compute the inverse of the block from the LU-factorization.
    Teuchos::ArrayRCP<double> work( m );
    lapack.GETRI( n, block.values(), lda, ipiv.getRawPtr(),
		  work.getRawPtr(), work.size(), &info );
    TEST_EQUALITY( info, 0 );
    TEST_EQUALITY( work[0], m );

    // Check the inversion against matlab.
    TEST_FLOATING_EQUALITY( block(0,0), -0.461356423424245, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(0,1), -0.060920073472551, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(0,2),  0.547244760641934, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(0,3),  0.412904055961420, 1.0e-14 );

    TEST_FLOATING_EQUALITY( block(1,0),  0.154767451798665, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(1,1), -0.056225122550555, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(1,2), -0.174451348828054, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(1,3), -0.055523340725809, 1.0e-14 );

    TEST_FLOATING_EQUALITY( block(2,0),  0.848746201780808, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(2,1),  0.045927762119214, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(2,2), -0.618485718805259, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(2,3), -0.415712965073367, 1.0e-14 );

    TEST_FLOATING_EQUALITY( block(3,0),  0.526232280383953, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(3,1), -0.069757566407458, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(3,2), -0.492378815120724, 1.0e-14 );
    TEST_FLOATING_EQUALITY( block(3,3), -0.505833501236923, 1.0e-14 );
}

//---------------------------------------------------------------------------//
// end tstBlockInversion.cpp
//---------------------------------------------------------------------------//
