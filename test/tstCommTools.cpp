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
 * \file   tstCommTools.cpp
 * \author Stuart Slattery
 * \brief  CommTools class unit tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <stack>
#include <sstream>
#include <stdexcept>

#include <MCLS_CommTools.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_ScalarTraits.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )	                   \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( type, name, float )      \
    TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( type, name, double )

//---------------------------------------------------------------------------//
// HELPER FUNCTIONS
//---------------------------------------------------------------------------//

// Get the default communicator.
template<class Ordinal>
Teuchos::RCP<const Teuchos::Comm<Ordinal> > getDefaultComm()
{
#ifdef HAVE_MPI
    return Teuchos::DefaultComm<Ordinal>::getComm();
#else
    return Teuchos::rcp(new Teuchos::SerialComm<Ordinal>() );
#endif
}

//---------------------------------------------------------------------------//
// Tests.
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( CommTools, ReduceSum, Scalar )
{
    Teuchos::RCP<const Teuchos::Comm<int> > comm = getDefaultComm<int>();

    int num_data = 10;
    Scalar value = 1.0;
    Teuchos::Array<Scalar> send_data( num_data, value );
    Teuchos::Array<Scalar> recv_data( num_data, 0.0 );

    int root = 0;
    MCLS::CommTools::reduceSum( comm, root, num_data,
				send_data.getRawPtr(), recv_data.getRawPtr() );

    Scalar red_value = (0 == comm->getRank()) ? comm->getSize() * value : 0.0;
    for ( int i = 0; i < num_data; ++i )
    {
	TEST_EQUALITY( red_value, recv_data[i] );
    }
}

UNIT_TEST_INSTANTIATION( CommTools, ReduceSum )

//---------------------------------------------------------------------------//
// end tstCommTools.cpp
//---------------------------------------------------------------------------//
