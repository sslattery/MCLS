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
 * \file tstTpetraHelpers.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra helper function tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_TpetraHelpers.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_BlockMap.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_VbrMatrix.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
// 
// These types are those enabled by Tpetra under explicit instantiation.
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )			           \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, double )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, double )

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( TpetraHelpers, CrsOffProcCols, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_rank = comm->getRank();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    Teuchos::RCP<MatrixType> A = 
	Tpetra::createCrsMatrix<Scalar,LO,GO>( map );

    Teuchos::Array<GO> global_columns( global_num_rows );
    for ( int j = 0; j < global_num_rows; ++j )
    {
	global_columns[j] = j;
    }
    Teuchos::Array<Scalar> values( global_num_rows, 1 );
    for ( int i = local_num_rows*comm_rank; i < local_num_rows*(comm_rank+1); ++i )
    {
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::Array<GO> off_proc_cols = 
	MCLS::TpetraMatrixHelpers<Scalar,LO,GO,MatrixType>::getOffProcColsAsRows( *A );

    int num_off_proc = off_proc_cols.size();
    int test_off_proc = local_num_rows*(comm_size-1);
    TEST_EQUALITY( num_off_proc, test_off_proc );
    
    Teuchos::Array<GO> local_columns( local_num_rows );
    for ( int i = local_num_rows*comm_rank, j = 0; 
	  i < local_num_rows*(comm_rank+1); 
	  ++i, ++j )
    {
	local_columns[j] = i;
    }
    Teuchos::Array<GO> test_columns( local_num_rows + global_num_rows );
    typename Teuchos::Array<GO>::iterator diff_it =
	std::set_difference( global_columns.begin(), global_columns.end(),
			     local_columns.begin(), local_columns.end(),
			     test_columns.begin() );
    test_columns.resize( std::distance(test_columns.begin(),diff_it) );
    int num_test_col = test_columns.size();
    TEST_EQUALITY( num_test_col, test_off_proc );
    
    std::sort( off_proc_cols.begin(), off_proc_cols.end() );
    for ( int i = 0; i < test_off_proc; ++i )
    {
	TEST_EQUALITY( off_proc_cols[i], test_columns[i] );
    }
}

UNIT_TEST_INSTANTIATION( TpetraHelpers, CrsOffProcCols )

//---------------------------------------------------------------------------//
// end tstTpetraHelpers.cpp
//---------------------------------------------------------------------------//

