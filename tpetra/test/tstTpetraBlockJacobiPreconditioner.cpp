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
 * \file tstTpetraBlockJacobiPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra block Jacobi preconditioning tests.
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

#include <MCLS_MatrixTraits.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_Preconditioner.hpp>
#include <MCLS_TpetraBlockJacobiPreconditioner.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( TpetraBlockJacobiPreconditioner, 1_block_matrix, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 4;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build a single block on each proc.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<Scalar> values_1( 4 );
    values_1[0] = 3.2;
    values_1[1] = -1.43;
    values_1[2] = 2.98;
    values_1[3] = 0.32;

    Teuchos::Array<Scalar> values_2( 4 );
    values_2[0] = -4.12;
    values_2[1] = -7.53;
    values_2[2] = 1.44;
    values_2[3] = -3.72;

    Teuchos::Array<Scalar> values_3( 4 );
    values_3[0] = 4.24;
    values_3[1] = -6.42;
    values_3[2] = 1.82;
    values_3[3] = 2.67;

    Teuchos::Array<Scalar> values_4( 4 );
    values_4[0] = -0.23;
    values_4[1] = 5.8;
    values_4[2] = 1.13;
    values_4[3] = -3.73;

    Teuchos::Array<Teuchos::Array<Scalar> > values( 4 );
    values[0] = values_1;
    values[1] = values_2;
    values[2] = values_3;
    values[3] = values_4;

    Teuchos::Array<GO> columns( 4 );
    columns[0] = local_num_rows*comm_rank;
    columns[1] = local_num_rows*comm_rank+1;
    columns[2] = local_num_rows*comm_rank+2;
    columns[3] = local_num_rows*comm_rank+3;

    GO global_row = 0;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	global_row = i + local_num_rows*comm_rank;
	A->insertGlobalValues( global_row, columns(), values[i]() );
    }
    A->fillComplete();

    // Build the preconditioner.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size", 4);
    Teuchos::RCP<MCLS::Preconditioner<MatrixType> > preconditioner = 
	Teuchos::rcp( 
	    new MCLS::TpetraBlockJacobiPreconditioner<Scalar,LO,GO>(plist) );
    preconditioner->setOperator( A );
    preconditioner->buildPreconditioner();
    Teuchos::RCP<const MatrixType> M = preconditioner->getLeftPreconditioner();

    // Check the preconditioner. Inverse block values from matlab.
    Teuchos::Array<GO> prec_cols(4);
    Teuchos::Array<Scalar> prec_vals(4);
    std::size_t num_entries = 0;

    global_row = local_num_rows*comm_rank;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns[0], prec_cols[0] );
    TEST_EQUALITY( columns[1], prec_cols[1] );
    TEST_EQUALITY( columns[2], prec_cols[2] );
    TEST_EQUALITY( columns[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0], -0.461356423424245, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.060920073472551, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2],  0.547244760641934, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3],  0.412904055961420, 1.0e-14 );

    global_row = local_num_rows*comm_rank+1;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns[0], prec_cols[0] );
    TEST_EQUALITY( columns[1], prec_cols[1] );
    TEST_EQUALITY( columns[2], prec_cols[2] );
    TEST_EQUALITY( columns[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.154767451798665, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.056225122550555, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.174451348828054, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.055523340725809, 1.0e-14 );

    global_row = local_num_rows*comm_rank+2;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns[0], prec_cols[0] );
    TEST_EQUALITY( columns[1], prec_cols[1] );
    TEST_EQUALITY( columns[2], prec_cols[2] );
    TEST_EQUALITY( columns[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.848746201780808, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1],  0.045927762119214, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.618485718805259, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.415712965073367, 1.0e-14 );

    global_row = local_num_rows*comm_rank+3;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns[0], prec_cols[0] );
    TEST_EQUALITY( columns[1], prec_cols[1] );
    TEST_EQUALITY( columns[2], prec_cols[2] );
    TEST_EQUALITY( columns[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.526232280383953, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.069757566407458, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.492378815120724, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.505833501236923, 1.0e-14 );
}

UNIT_TEST_INSTANTIATION( TpetraBlockJacobiPreconditioner, 1_block_matrix )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( TpetraBlockJacobiPreconditioner, 2_block_matrix, LO, GO, Scalar )
{
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 8;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build a single block on each proc.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<Scalar> values_1( 4 );
    values_1[0] = 3.2;
    values_1[1] = -1.43;
    values_1[2] = 2.98;
    values_1[3] = 0.32;

    Teuchos::Array<Scalar> values_2( 4 );
    values_2[0] = -4.12;
    values_2[1] = -7.53;
    values_2[2] = 1.44;
    values_2[3] = -3.72;

    Teuchos::Array<Scalar> values_3( 4 );
    values_3[0] = 4.24;
    values_3[1] = -6.42;
    values_3[2] = 1.82;
    values_3[3] = 2.67;

    Teuchos::Array<Scalar> values_4( 4 );
    values_4[0] = -0.23;
    values_4[1] = 5.8;
    values_4[2] = 1.13;
    values_4[3] = -3.73;

    Teuchos::Array<Teuchos::Array<Scalar> > values( 4 );
    values[0] = values_1;
    values[1] = values_2;
    values[2] = values_3;
    values[3] = values_4;

    Teuchos::Array<GO> columns_1( 4 );
    columns_1[0] = local_num_rows*comm_rank;
    columns_1[1] = local_num_rows*comm_rank+1;
    columns_1[2] = local_num_rows*comm_rank+2;
    columns_1[3] = local_num_rows*comm_rank+3;

    Teuchos::Array<GO> columns_2( 4 );
    columns_2[0] = local_num_rows*comm_rank+4;
    columns_2[1] = local_num_rows*comm_rank+5;
    columns_2[2] = local_num_rows*comm_rank+6;
    columns_2[3] = local_num_rows*comm_rank+7;

    GO global_row = 0;

    // block 1
    for ( int i = 0; i < 4; ++i )
    {
	global_row = i + local_num_rows*comm_rank;
	A->insertGlobalValues( global_row, columns_1(), values[i]() );
    }
    // block 2
    for ( int i = 4; i < local_num_rows; ++i )
    {
	global_row = i + local_num_rows*comm_rank;
	A->insertGlobalValues( global_row, columns_2(), values[i-4]() );
    }
    A->fillComplete();

    // Build the preconditioner.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size", 4);
    Teuchos::RCP<MCLS::Preconditioner<MatrixType> > preconditioner = 
	Teuchos::rcp( 
	    new MCLS::TpetraBlockJacobiPreconditioner<Scalar,LO,GO>(plist) );
    preconditioner->setOperator( A );
    preconditioner->buildPreconditioner();
    Teuchos::RCP<const MatrixType> M = preconditioner->getLeftPreconditioner();

    // Check the preconditioner. Inverse block values from matlab.
    Teuchos::Array<GO> prec_cols(4);
    Teuchos::Array<Scalar> prec_vals(4);
    std::size_t num_entries = 0;

    // block 1
    global_row = local_num_rows*comm_rank;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_1[0], prec_cols[0] );
    TEST_EQUALITY( columns_1[1], prec_cols[1] );
    TEST_EQUALITY( columns_1[2], prec_cols[2] );
    TEST_EQUALITY( columns_1[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0], -0.461356423424245, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.060920073472551, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2],  0.547244760641934, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3],  0.412904055961420, 1.0e-14 );

    global_row = local_num_rows*comm_rank+1;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_1[0], prec_cols[0] );
    TEST_EQUALITY( columns_1[1], prec_cols[1] );
    TEST_EQUALITY( columns_1[2], prec_cols[2] );
    TEST_EQUALITY( columns_1[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.154767451798665, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.056225122550555, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.174451348828054, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.055523340725809, 1.0e-14 );

    global_row = local_num_rows*comm_rank+2;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_1[0], prec_cols[0] );
    TEST_EQUALITY( columns_1[1], prec_cols[1] );
    TEST_EQUALITY( columns_1[2], prec_cols[2] );
    TEST_EQUALITY( columns_1[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.848746201780808, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1],  0.045927762119214, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.618485718805259, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.415712965073367, 1.0e-14 );

    global_row = local_num_rows*comm_rank+3;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_1[0], prec_cols[0] );
    TEST_EQUALITY( columns_1[1], prec_cols[1] );
    TEST_EQUALITY( columns_1[2], prec_cols[2] );
    TEST_EQUALITY( columns_1[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.526232280383953, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.069757566407458, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.492378815120724, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.505833501236923, 1.0e-14 );

    // block 2
    global_row = local_num_rows*comm_rank+4;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_2[0], prec_cols[0] );
    TEST_EQUALITY( columns_2[1], prec_cols[1] );
    TEST_EQUALITY( columns_2[2], prec_cols[2] );
    TEST_EQUALITY( columns_2[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0], -0.461356423424245, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.060920073472551, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2],  0.547244760641934, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3],  0.412904055961420, 1.0e-14 );

    global_row = local_num_rows*comm_rank+5;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_2[0], prec_cols[0] );
    TEST_EQUALITY( columns_2[1], prec_cols[1] );
    TEST_EQUALITY( columns_2[2], prec_cols[2] );
    TEST_EQUALITY( columns_2[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.154767451798665, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.056225122550555, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.174451348828054, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.055523340725809, 1.0e-14 );

    global_row = local_num_rows*comm_rank+6;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_2[0], prec_cols[0] );
    TEST_EQUALITY( columns_2[1], prec_cols[1] );
    TEST_EQUALITY( columns_2[2], prec_cols[2] );
    TEST_EQUALITY( columns_2[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.848746201780808, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1],  0.045927762119214, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.618485718805259, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.415712965073367, 1.0e-14 );

    global_row = local_num_rows*comm_rank+7;
    MT::getGlobalRowCopy( *M, global_row, prec_cols(), prec_vals(), num_entries );
    TEST_EQUALITY( num_entries, 4 );
    TEST_EQUALITY( columns_2[0], prec_cols[0] );
    TEST_EQUALITY( columns_2[1], prec_cols[1] );
    TEST_EQUALITY( columns_2[2], prec_cols[2] );
    TEST_EQUALITY( columns_2[3], prec_cols[3] );
    TEST_FLOATING_EQUALITY( prec_vals[0],  0.526232280383953, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[1], -0.069757566407458, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[2], -0.492378815120724, 1.0e-14 );
    TEST_FLOATING_EQUALITY( prec_vals[3], -0.505833501236923, 1.0e-14 );
}

UNIT_TEST_INSTANTIATION( TpetraBlockJacobiPreconditioner, 2_block_matrix )

//---------------------------------------------------------------------------//
// end tstTpetraBlockJacobiPreconditioner.cpp
//---------------------------------------------------------------------------//

