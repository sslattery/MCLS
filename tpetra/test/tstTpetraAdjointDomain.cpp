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
 * \file tstTpetraAdjointDomain.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra AdjointDomain tests.
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
#include <random>

#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_AdjointHistory.hpp>
#include <MCLS_AdjointTally.hpp>
#include <MCLS_Events.hpp>
#include <MCLS_PRNG.hpp>
#include <MCLS_MatrixTraits.hpp>

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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef std::mt19937 rng_type;
    typedef MCLS::AdjointDomain<VectorType,MatrixType,rng_type> DomainType;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef typename DomainType::HistoryType history_type;
    typedef typename DomainType::TallyType tally_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<TallyType, tally_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( AdjointDomain, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, NoOverlap, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef std::mt19937 rng_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> first_columns( 1, 0 );
    Teuchos::Array<Scalar> first_values( 1, 2.0 );
    A->insertGlobalValues( 0, first_columns(), first_values() );
    Teuchos::Array<GO> global_columns( 2 );
    Teuchos::Array<Scalar> values( 2 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	values[0] = 2;
	values[1] = 3;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> domain( A_T, x, plist );

    // Check the tally.
    Scalar x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i-comm_rank*local_num_rows, x_val );
	    history.live();
	    tally->tallyHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const Scalar> x_view = VT::view( *x );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator x_view_iterator;
    for ( x_view_iterator = x_view.begin();
	  x_view_iterator != x_view.end();
	  ++x_view_iterator )
    {
	TEST_EQUALITY( *x_view_iterator, x_val );
    }

    // Check the boundary.
    if ( comm_rank == 0 && comm_size == 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == 0 && comm_size > 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == comm_size - 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }
    else
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }

    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    TEST_ASSERT( domain.isGlobalState(i) );
	}
	else
	{
	    TEST_ASSERT( !domain.isGlobalState(i) );
	}
    }
}

UNIT_TEST_INSTANTIATION( AdjointDomain, NoOverlap )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, PackUnpack, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef std::mt19937 rng_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> first_columns( 1, 0 );
    Teuchos::Array<Scalar> first_values( 1, 2.0 );
    A->insertGlobalValues( 0, first_columns(), first_values() );
    Teuchos::Array<GO> global_columns( 2 );
    Teuchos::Array<Scalar> values( 2 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	values[0] = 2;
	values[1] = 3;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> primary_domain( A_T, x, plist );

    // Pack the domain into a buffer.
    Teuchos::Array<char> domain_buffer = primary_domain.pack();

    // Unpack the domain to make a new one for testing.
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> domain( domain_buffer, comm );

    // Check the tally.
    Scalar x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setBaseVector( x );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i-comm_rank*local_num_rows, x_val );
	    history.live();
	    tally->tallyHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const Scalar> x_view = VT::view( *x );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator x_view_iterator;
    for ( x_view_iterator = x_view.begin();
	  x_view_iterator != x_view.end();
	  ++x_view_iterator )
    {
	TEST_EQUALITY( *x_view_iterator, x_val );
    }

    // Check the boundary.
    if ( comm_rank == 0 && comm_size == 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == 0 && comm_size > 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == comm_size - 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }
    else
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }

    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    TEST_ASSERT( domain.isGlobalState(i) );
	}
	else
	{
	    TEST_ASSERT( !domain.isGlobalState(i) );
	}
    }
}

UNIT_TEST_INSTANTIATION( AdjointDomain, PackUnpack )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, SomeOverlap, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef std::mt19937 rng_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> first_columns( 1, 0 );
    Teuchos::Array<Scalar> first_values( 1, 2.0 );
    A->insertGlobalValues( 0, first_columns(), first_values() );
    Teuchos::Array<GO> global_columns( 2 );
    Teuchos::Array<Scalar> values( 2 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	values[0] = 2;
	values[1] = 3;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> domain( A_T, x, plist );

    // Check the tally.
    Scalar x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i-comm_rank*local_num_rows, x_val );
	    history.live();
	    tally->tallyHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const Scalar> x_view = VT::view( *x );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	if ( comm_rank == 0 || i > 1 )
	{
	    TEST_EQUALITY( x_view[i], x_val );
	}
	else
	{
	    TEST_EQUALITY( x_view[i], 2*x_val );
	}
    }

    // Check the boundary.
    if ( comm_rank == 0 && comm_size == 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == 0 && comm_size > 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(2+local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == comm_size - 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }
    else
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(2+local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }

    if ( comm_rank == comm_size-1 )
    {
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	    {
		TEST_ASSERT( domain.isGlobalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isGlobalState(i) );
	    }
	}
    }
    else
    {
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	    {
		TEST_ASSERT( domain.isGlobalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isGlobalState(i) );
	    }
	}
    }
}

UNIT_TEST_INSTANTIATION( AdjointDomain, SomeOverlap )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, PackUnpackSomeOverlap, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef std::mt19937 rng_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> first_columns( 1, 0 );
    Teuchos::Array<Scalar> first_values( 1, 2.0 );
    A->insertGlobalValues( 0, first_columns(), first_values() );
    Teuchos::Array<GO> global_columns( 2 );
    Teuchos::Array<Scalar> values( 2 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	values[0] = 2;
	values[1] = 3;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> primary_domain( A_T, x, plist );

    // Pack the domain into a buffer.
    Teuchos::Array<char> domain_buffer = primary_domain.pack();

    // Unpack the domain to make a new one for testing.
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> domain( domain_buffer, comm );

    // Check the tally.
    Scalar x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setBaseVector( x );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i-comm_rank*local_num_rows, x_val );
	    history.live();
	    tally->tallyHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const Scalar> x_view = VT::view( *x );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	if ( comm_rank == 0 || i > 1 )
	{
	    TEST_EQUALITY( x_view[i], x_val );
	}
	else
	{
	    TEST_EQUALITY( x_view[i], 2*x_val );
	}
    }

    // Check the boundary.
    if ( comm_rank == 0 && comm_size == 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == 0 && comm_size > 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(2+local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 0 );
    }
    else if ( comm_rank == comm_size - 1 )
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }
    else
    {
	TEST_EQUALITY( domain.numSendNeighbors(), 1 );
	TEST_EQUALITY( domain.sendNeighborRank(0), comm_rank+1 );
	TEST_EQUALITY( domain.owningNeighbor(2+local_num_rows*(comm_rank+1)), 0 );
	TEST_EQUALITY( domain.numReceiveNeighbors(), 1 );
	TEST_EQUALITY( domain.receiveNeighborRank(0), comm_rank-1 );
    }

    if ( comm_rank == comm_size-1 )
    {
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	    {
		TEST_ASSERT( domain.isGlobalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isGlobalState(i) );
	    }
	}
    }
    else
    {
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	    {
		TEST_ASSERT( domain.isGlobalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isGlobalState(i) );
	    }
	}
    }
}

UNIT_TEST_INSTANTIATION( AdjointDomain, PackUnpackSomeOverlap )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, Transition, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef std::mt19937 rng_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    values[0] = -0.5;
    global_columns[0] = global_num_rows-1;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> domain( A_T, x, plist );

    // Process a history transition in the domain.
    Teuchos::RCP<MCLS::PRNG<rng_type> > rng = Teuchos::rcp(
	new MCLS::PRNG<rng_type>( comm->getRank() ) );
    domain.setRNG( rng );
    double weight = 3.0; 
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i-comm_rank*local_num_rows, weight );
	    history.live();
	    history.setEvent( MCLS::Event::TRANSITION );
	    domain.processTransition( history );

	    TEST_EQUALITY( history.globalState(), i+1 );
	    TEST_EQUALITY( history.weight(), weight*comm_size / 2 );
	}
    }
}

UNIT_TEST_INSTANTIATION( AdjointDomain, Transition )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( AdjointDomain, Diagonal, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<GO> HistoryType;
    typedef std::mt19937 rng_type;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
    Teuchos::Array<GO> global_columns( 1 );
    Teuchos::Array<Scalar> values( 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	values[0] = 3.0;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    MCLS::AdjointDomain<VectorType,MatrixType,rng_type> domain( A, x, plist );

    // Process a history transition in the domain.
    Teuchos::RCP<MCLS::PRNG<rng_type> > rng = Teuchos::rcp(
	new MCLS::PRNG<rng_type>( comm->getRank() ) );
    domain.setRNG( rng );
    double weight = 3.0; 
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i-comm_rank*local_num_rows, weight );
	    history.live();
	    history.setEvent( MCLS::Event::TRANSITION );
	    domain.processTransition( history );

	    TEST_EQUALITY( history.globalState(), i );
	    TEST_EQUALITY( history.weight(), -weight*(comm_size*3-1) );
	}
    }
}

UNIT_TEST_INSTANTIATION( AdjointDomain, Diagonal )

//---------------------------------------------------------------------------//
// end tstTpetraAdjointDomain.cpp
//---------------------------------------------------------------------------//

