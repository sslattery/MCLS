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
 * \file tstEpetraBlockJacobiPreconditioner.cpp
 * \author Stuart R. Slattery
 * \brief Epetra block Jacobi preconditioning tests.
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
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_Preconditioner.hpp>
#include <MCLS_EpetraBlockJacobiPreconditioner.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Epetra_SerialDenseMatrix.h>
#include <Epetra_Map.h>
#include <Epetra_BlockMap.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_VbrMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#endif

//---------------------------------------------------------------------------//
// Helper functions.
//---------------------------------------------------------------------------//

Teuchos::RCP<Epetra_Comm> getEpetraComm( 
    const Teuchos::RCP<const Teuchos::Comm<int> >& comm )
{
#ifdef HAVE_MPI
    Teuchos::RCP< const Teuchos::MpiComm<int> > mpi_comm = 
	Teuchos::rcp_dynamic_cast< const Teuchos::MpiComm<int> >( comm );
    Teuchos::RCP< const Teuchos::OpaqueWrapper<MPI_Comm> > opaque_comm = 
	mpi_comm->getRawMpiComm();
    return Teuchos::rcp( new Epetra_MpiComm( (*opaque_comm)() ) );
#else
    return Teuchos::rcp( new Epetra_SerialComm() );
#endif
}

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( EpetraBlockJacobiPreconditioner, 1_block_matrix )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 4;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build a single block on each proc.
    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<double> values_1( 4 );
    values_1[0] = 3.2;
    values_1[1] = -1.43;
    values_1[2] = 2.98;
    values_1[3] = 0.32;

    Teuchos::Array<double> values_2( 4 );
    values_2[0] = -4.12;
    values_2[1] = -7.53;
    values_2[2] = 1.44;
    values_2[3] = -3.72;

    Teuchos::Array<double> values_3( 4 );
    values_3[0] = 4.24;
    values_3[1] = -6.42;
    values_3[2] = 1.82;
    values_3[3] = 2.67;

    Teuchos::Array<double> values_4( 4 );
    values_4[0] = -0.23;
    values_4[1] = 5.8;
    values_4[2] = 1.13;
    values_4[3] = -3.73;

    Teuchos::Array<Teuchos::Array<double> > values( 4 );
    values[0] = values_1;
    values[1] = values_2;
    values[2] = values_3;
    values[3] = values_4;

    Teuchos::Array<int> columns( 4 );
    columns[0] = local_num_rows*comm_rank;
    columns[1] = local_num_rows*comm_rank+1;
    columns[2] = local_num_rows*comm_rank+2;
    columns[3] = local_num_rows*comm_rank+3;

    int global_row = 0;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	global_row = i + local_num_rows*comm_rank;
	A->InsertGlobalValues( global_row, columns.size(), 
			       values[i].getRawPtr(), columns.getRawPtr() );
    }
    A->FillComplete();

    // Build the preconditioner.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size", 4);
    Teuchos::RCP<MCLS::Preconditioner<MatrixType> > preconditioner = 
	Teuchos::rcp( new MCLS::EpetraBlockJacobiPreconditioner(plist) );
    preconditioner->setOperator( A );
    preconditioner->buildPreconditioner();
    Teuchos::RCP<const MatrixType> M = preconditioner->getLeftPreconditioner();

    // Check the preconditioner. Inverse block values from matlab.
    Teuchos::Array<int> prec_cols(4);
    Teuchos::Array<double> prec_vals(4);
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

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( EpetraBlockJacobiPreconditioner, vbr_1_block_matrix )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 4;
    int global_num_rows = comm_size;
    Teuchos::RCP<Epetra_BlockMap> map = Teuchos::rcp(
	new Epetra_BlockMap( global_num_rows, 1, 4, 0, *epetra_comm ) );

    Teuchos::Array<int> columns( 4 );
    columns[0] = local_num_rows*comm_rank;
    columns[1] = local_num_rows*comm_rank+1;
    columns[2] = local_num_rows*comm_rank+2;
    columns[3] = local_num_rows*comm_rank+3;

    Teuchos::Array<int> local_cols( 1, comm_rank );

    // Build a single block on each proc.
    Teuchos::RCP<Epetra_VbrMatrix> A = 
	Teuchos::rcp( new Epetra_VbrMatrix( Copy, *map, 1 ) );

    // Build a 4x4 block.
    int m = 4;
    int n = 4;
    Epetra_SerialDenseMatrix block( m, n );

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

    int errval = 0;
    errval = A->BeginInsertGlobalValues( comm_rank, 1, local_cols.getRawPtr() );
    TEST_EQUALITY( errval, 0 );
    errval = A->SubmitBlockEntry(block);
    TEST_EQUALITY( errval, 0 );
    errval = A->EndSubmitEntries();
    TEST_EQUALITY( errval, 0 );
    errval = A->FillComplete();

    // Build the preconditioner.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size", 4);
    Teuchos::RCP<MCLS::Preconditioner<MatrixType> > preconditioner = 
	Teuchos::rcp( new MCLS::EpetraBlockJacobiPreconditioner(plist) );
    preconditioner->setOperator( A );
    preconditioner->buildPreconditioner();
    Teuchos::RCP<const MatrixType> M = preconditioner->getLeftPreconditioner();

    // Check the preconditioner. Inverse block values from matlab.
    Teuchos::Array<int> prec_cols(4);
    Teuchos::Array<double> prec_vals(4);
    std::size_t num_entries = 0;

    int global_row = local_num_rows*comm_rank;
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

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( EpetraBlockJacobiPreconditioner, 2_block_matrix )
{
    typedef Epetra_RowMatrix MatrixType;
    typedef Epetra_Vector VectorType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 8;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build a single block on each proc.
    Teuchos::RCP<Epetra_CrsMatrix> A = 
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );

    Teuchos::Array<double> values_1( 8 );
    values_1[0] = 3.2;
    values_1[1] = -1.43;
    values_1[2] = 2.98;
    values_1[3] = 0.32;
    values_1[4] = -1.3;
    values_1[5] = -1.3;
    values_1[6] = -1.3;
    values_1[7] = -1.3;

    Teuchos::Array<double> values_2( 8 );
    values_2[0] = -4.12;
    values_2[1] = -7.53;
    values_2[2] = 1.44;
    values_2[3] = -3.72;
    values_2[4] = -1.3;
    values_2[5] = -1.3;
    values_2[6] = -1.3;
    values_2[7] = -1.3;

    Teuchos::Array<double> values_3( 8 );
    values_3[0] = 4.24;
    values_3[1] = -6.42;
    values_3[2] = 1.82;
    values_3[3] = 2.67;
    values_3[4] = -1.3;
    values_3[5] = -1.3;
    values_3[6] = -1.3;
    values_3[7] = -1.3;

    Teuchos::Array<double> values_4( 8 );
    values_4[0] = -0.23;
    values_4[1] = 5.8;
    values_4[2] = 1.13;
    values_4[3] = -3.73;
    values_4[4] = -1.3;
    values_4[5] = -1.3;
    values_4[6] = -1.3;
    values_4[7] = -1.3;

    Teuchos::Array<Teuchos::Array<double> > values( 4 );
    values[0] = values_1;
    values[1] = values_2;
    values[2] = values_3;
    values[3] = values_4;

    Teuchos::Array<int> columns_1( 8 );
    columns_1[0] = local_num_rows*comm_rank;
    columns_1[1] = local_num_rows*comm_rank+1;
    columns_1[2] = local_num_rows*comm_rank+2;
    columns_1[3] = local_num_rows*comm_rank+3;
    columns_1[4] = local_num_rows*comm_rank+4;
    columns_1[5] = local_num_rows*comm_rank+5;
    columns_1[6] = local_num_rows*comm_rank+6;
    columns_1[7] = local_num_rows*comm_rank+7;

    Teuchos::Array<int> columns_2( 8 );
    columns_2[0] = local_num_rows*comm_rank+4;
    columns_2[1] = local_num_rows*comm_rank+5;
    columns_2[2] = local_num_rows*comm_rank+6;
    columns_2[3] = local_num_rows*comm_rank+7;
    columns_2[4] = local_num_rows*comm_rank;
    columns_2[5] = local_num_rows*comm_rank+1;
    columns_2[6] = local_num_rows*comm_rank+2;
    columns_2[7] = local_num_rows*comm_rank+3;

    int global_row = 0;

    // block 1
    for ( int i = 0; i < 4; ++i )
    {
	global_row = i + local_num_rows*comm_rank;
	A->InsertGlobalValues( global_row, columns_1.size(), 
			       values[i].getRawPtr(), columns_1.getRawPtr() );
    }
    // block 2
    for ( int i = 4; i < local_num_rows; ++i )
    {
	global_row = i + local_num_rows*comm_rank;
	A->InsertGlobalValues( global_row, columns_2.size(), 
			       values[i-4].getRawPtr(), columns_2.getRawPtr() );
    }
    A->FillComplete();

    // Build the preconditioner.
    Teuchos::RCP<Teuchos::ParameterList> plist = Teuchos::parameterList();
    plist->set<int>("Jacobi Block Size", 4);
    Teuchos::RCP<MCLS::Preconditioner<MatrixType> > preconditioner = 
	Teuchos::rcp( new MCLS::EpetraBlockJacobiPreconditioner(plist) );
    preconditioner->setOperator( A );
    preconditioner->buildPreconditioner();
    Teuchos::RCP<const MatrixType> M = preconditioner->getLeftPreconditioner();

    // Check the preconditioner. Inverse block values from matlab.
    Teuchos::Array<int> prec_cols(4);
    Teuchos::Array<double> prec_vals(4);
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

//---------------------------------------------------------------------------//
// end tstEpetraBlockJacobiPreconditioner.cpp
//---------------------------------------------------------------------------//

