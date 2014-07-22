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
 * \file tstTpetraUniformForwardSource.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra UniformForwardSource tests.
 */
//---------------------------------------------------------------------------//

#include <stack>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_UniformForwardSource.hpp>
#include <MCLS_ForwardDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_ForwardHistory.hpp>
#include <MCLS_ForwardTally.hpp>
#include <MCLS_Events.hpp>
#include <MCLS_RNGControl.hpp>

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

//---------------------------------------------------------------------------//
// Instantiation macro. 
// 
// These types are those enabled by Tpetra under explicit instantiation. I
// have removed scalar types that are not floating point
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )			           \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, int, double )   \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, double )

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformForwardSource, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::ForwardDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::ForwardHistory<GO> HistoryType;

    typedef MCLS::UniformForwardSource<DomainType> SourceType;
    typedef typename SourceType::HistoryType history_type;
    typedef typename SourceType::VectorType vector_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<VectorType, vector_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( UniformForwardSource, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformForwardSource, nh_not_set, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<GO> HistoryType;
    typedef MCLS::ForwardDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear system.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *b, 1.0 );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A_T, x, plist ) );

    // History setup.
    Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	new MCLS::RNGControl( 3939294 ) );
    HistoryType::setByteSize( control->getSize() );

    // Create the forward source with default values.
    double cutoff = 1.0e-8;
    plist.set<double>("Weight Cutoff", cutoff );
    MCLS::UniformForwardSource<DomainType> 
	source( b, domain, control, comm, comm->getSize(), comm->getRank(), plist );
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numToTransport(), 0 );
    TEST_EQUALITY( source.numToTransportInSet(), global_num_rows );
    TEST_EQUALITY( source.numRequested(), global_num_rows );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), 0 );

    // Build the source.
    source.buildSource();
    TEST_ASSERT( !source.empty() );
    TEST_EQUALITY( source.numToTransport(), local_num_rows );
    TEST_EQUALITY( source.numToTransportInSet(), global_num_rows );
    TEST_EQUALITY( source.numRequested(), global_num_rows );
    TEST_EQUALITY( source.numLeft(), local_num_rows );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), comm->getSize() );
    TEST_EQUALITY( source.sourceWeight(), 1.0 );

    // Sample the source.
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( !source.empty() );
	TEST_EQUALITY( source.numLeft(), local_num_rows-i );
	TEST_EQUALITY( source.numEmitted(), i );

	Teuchos::RCP<HistoryType> history = source.getHistory();

	TEST_EQUALITY( history->weight(), 1.0 );
	TEST_ASSERT( domain->isLocalState( history->state() ) );
	TEST_ASSERT( history->alive() );
	TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
    }
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), local_num_rows );
}

UNIT_TEST_INSTANTIATION( UniformForwardSource, nh_not_set )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformForwardSource, PackUnpack, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<GO> HistoryType;
    typedef MCLS::ForwardDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear system.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *b, -1.0 );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A_T, x, plist ) );

    // History setup.
    Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	new MCLS::RNGControl( 3939294 ) );
    HistoryType::setByteSize( control->getSize() );

    // Create the forward source with default values.
    double cutoff = 1.0e-8;
    plist.set<double>("Weight Cutoff", cutoff );
    MCLS::UniformForwardSource<DomainType> 
	primary_source( b, domain, control, comm, comm->getSize(), 
			comm->getRank(), plist );

    // Pack and unpack the source.
    Teuchos::Array<char> source_buffer = primary_source.pack();
    MCLS::UniformForwardSource<DomainType> 
	source( source_buffer, domain, control, 
		comm, comm->getSize(), comm->getRank() );
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numToTransport(), 0 );
    TEST_EQUALITY( source.numToTransportInSet(), global_num_rows );
    TEST_EQUALITY( source.numRequested(), global_num_rows );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), 0 );

    // Build the source.
    source.buildSource();
    TEST_ASSERT( !source.empty() );
    TEST_EQUALITY( source.numToTransport(), local_num_rows );
    TEST_EQUALITY( source.numToTransportInSet(), global_num_rows );
    TEST_EQUALITY( source.numRequested(), global_num_rows );
    TEST_EQUALITY( source.numLeft(), local_num_rows );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), comm->getSize() );

    // Sample the source.
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_ASSERT( !source.empty() );
	TEST_EQUALITY( source.numLeft(), local_num_rows-i );
	TEST_EQUALITY( source.numEmitted(), i );

	Teuchos::RCP<HistoryType> history = source.getHistory();

	TEST_EQUALITY( history->weight(), 1.0 );
	TEST_ASSERT( domain->isLocalState( history->state() ) );
	TEST_ASSERT( history->alive() );
	TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
    }
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), local_num_rows );
}

UNIT_TEST_INSTANTIATION( UniformForwardSource, PackUnpack )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformForwardSource, nh_set, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<GO> HistoryType;
    typedef MCLS::ForwardDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear system.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *b, -1.0 );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A_T, x, plist ) );

    // History setup.
    Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	new MCLS::RNGControl( 3939294 ) );
    HistoryType::setByteSize( control->getSize() );

    // Create the forward source with a set number of histories.
    int mult = 10;
    double cutoff = 1.0e-8;
    plist.set<int>("Set Number of Histories", mult*global_num_rows);
    plist.set<double>("Weight Cutoff", cutoff);
    MCLS::UniformForwardSource<DomainType> 
	source( b, domain, control, comm, comm->getSize(), comm->getRank(), plist );
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numToTransport(), 0 );
    TEST_EQUALITY( source.numToTransportInSet(), mult*global_num_rows );
    TEST_EQUALITY( source.numRequested(), mult*global_num_rows );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), 0 );

    // Build the source.
    source.buildSource();
    TEST_ASSERT( !source.empty() );
    TEST_EQUALITY( source.numToTransport(), mult*local_num_rows );
    TEST_EQUALITY( source.numToTransportInSet(), mult*global_num_rows );
    TEST_EQUALITY( source.numRequested(), mult*global_num_rows );
    TEST_EQUALITY( source.numLeft(), mult*local_num_rows );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), comm->getSize() );

    // Sample the source.
    for ( int i = 0; i < mult*local_num_rows; ++i )
    {
	TEST_ASSERT( !source.empty() );
	TEST_EQUALITY( source.numLeft(), mult*local_num_rows-i );
	TEST_EQUALITY( source.numEmitted(), i );

	Teuchos::RCP<HistoryType> history = source.getHistory();

	TEST_EQUALITY( history->weight(), 1.0 );
	TEST_ASSERT( domain->isLocalState( history->state() ) );
	TEST_ASSERT( history->alive() );
	TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
    }
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), mult*local_num_rows );
}

UNIT_TEST_INSTANTIATION( UniformForwardSource, nh_set )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( UniformForwardSource, nh_set_pu, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<GO> HistoryType;
    typedef MCLS::ForwardDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear system.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *b, -1.0 );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A_T, x, plist ) );

    // History setup.
    Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	new MCLS::RNGControl( 3939294 ) );
    HistoryType::setByteSize( control->getSize() );

    // Create the forward source with a set number of histories.
    int mult = 10;
    double cutoff = 1.0e-8;
    plist.set<int>("Set Number of Histories", mult*global_num_rows);
    plist.set<double>("Weight Cutoff", cutoff);
    MCLS::UniformForwardSource<DomainType> 
	primary_source( b, domain, control, comm, comm->getSize(), 
			comm->getRank(), plist );

    // Pack and unpack the source.
    Teuchos::Array<char> source_buffer = primary_source.pack();
    MCLS::UniformForwardSource<DomainType> 
	source( source_buffer, domain, control, 
		comm, comm->getSize(), comm->getRank() );

    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numToTransport(), 0 );
    TEST_EQUALITY( source.numToTransportInSet(), mult*global_num_rows );
    TEST_EQUALITY( source.numRequested(), mult*global_num_rows );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), 0 );

    // Build the source.
    source.buildSource();
    TEST_ASSERT( !source.empty() );
    TEST_EQUALITY( source.numToTransport(), mult*local_num_rows );
    TEST_EQUALITY( source.numToTransportInSet(), mult*global_num_rows );
    TEST_EQUALITY( source.numRequested(), mult*global_num_rows );
    TEST_EQUALITY( source.numLeft(), mult*local_num_rows );
    TEST_EQUALITY( source.numEmitted(), 0 );
    TEST_EQUALITY( source.numStreams(), comm->getSize() );

    // Sample the source.
    for ( int i = 0; i < mult*local_num_rows; ++i )
    {
	TEST_ASSERT( !source.empty() );
	TEST_EQUALITY( source.numLeft(), mult*local_num_rows-i );
	TEST_EQUALITY( source.numEmitted(), i );

	Teuchos::RCP<HistoryType> history = source.getHistory();

	TEST_EQUALITY( history->weight(), 1.0 );
	TEST_ASSERT( domain->isLocalState( history->state() ) );
	TEST_ASSERT( history->alive() );
	TEST_ASSERT( VT::isGlobalRow( *x, history->state() ) );
    }
    TEST_ASSERT( source.empty() );
    TEST_EQUALITY( source.numLeft(), 0 );
    TEST_EQUALITY( source.numEmitted(), mult*local_num_rows );
}

UNIT_TEST_INSTANTIATION( UniformForwardSource, nh_set_pu )

//---------------------------------------------------------------------------//
// end tstTpetraUniformForwardSource.cpp
//---------------------------------------------------------------------------//

