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
 * \file tstEpetraForwardDomain.cpp
 * \author Stuart R. Slattery
 * \brief Epetra ForwardDomain tests.
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

#include <MCLS_ForwardDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>
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
#include <Teuchos_OpaqueWrapper.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_CrsMatrix.h>
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
TEUCHOS_UNIT_TEST( ForwardDomain, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::ForwardDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;
    typedef DomainType::HistoryType history_type;
    typedef DomainType::TallyType tally_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<TallyType, tally_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardDomain, NoOverlap )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> first_columns( 1, 0 );
    Teuchos::Array<double> first_values( 1, 2.0 );
    A->InsertGlobalValues( 0, first_columns().size(), 
                           &first_values[0], &first_columns[0] );
    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 2;
	values[1] = 3;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-1;
    A->InsertGlobalValues( global_num_rows-1, 1,
                           &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    MCLS::ForwardDomain<VectorType,MatrixType> domain( B, x, plist );

    // Check the tally.
    Teuchos::RCP<VectorType> y = 
        VT::createFromRows( comm, domain.localStates()() );
    double y_val = 5.0;
    VT::putScalar( *y, y_val );
    double x_val = 2.0;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setSource( y );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, x_val );
	    history.live();
	    tally->tallyHistory( history );
            history.setEvent( MCLS::Event::CUTOFF );
            history.kill();
            tally->postProcessHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const double> x_view = VT::view( *x );
    Teuchos::ArrayRCP<const double>::const_iterator x_view_iterator;
    for ( x_view_iterator = x_view.begin();
	  x_view_iterator != x_view.end();
	  ++x_view_iterator )
    {
	TEST_EQUALITY( *x_view_iterator, x_val*y_val );
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
	    TEST_ASSERT( domain.isLocalState(i) );
	}
	else
	{
	    TEST_ASSERT( !domain.isLocalState(i) );
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardDomain, PackUnpack )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> first_columns( 1, 0 );
    Teuchos::Array<double> first_values( 1, 2.0 );
    A->InsertGlobalValues( 0, first_columns().size(), 
                           &first_values[0], &first_columns[0] );
    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 2;
	values[1] = 3;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-1;
    A->InsertGlobalValues( global_num_rows-1, 1,
                           &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 0 );
    MCLS::ForwardDomain<VectorType,MatrixType> primary_domain( B, x, plist );

    // Pack the domain into a buffer.
    Teuchos::Array<char> domain_buffer = primary_domain.pack();

    // Unpack the domain to make a new one for testing.
    MCLS::ForwardDomain<VectorType,MatrixType> domain( domain_buffer, comm );

    // Check the tally.
    Teuchos::RCP<VectorType> y = 
        VT::createFromRows( comm, domain.localStates()() );
    double y_val = 5.0;
    VT::putScalar( *y, y_val );
    double x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setBaseVector( x );
    tally->setSource( y );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, x_val );
	    history.live();
	    tally->tallyHistory( history );
            history.setEvent( MCLS::Event::CUTOFF );
            history.kill();
            tally->postProcessHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const double> x_view = VT::view( *x );
    Teuchos::ArrayRCP<const double>::const_iterator x_view_iterator;
    for ( x_view_iterator = x_view.begin();
	  x_view_iterator != x_view.end();
	  ++x_view_iterator )
    {
	TEST_EQUALITY( *x_view_iterator, x_val*y_val );
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
	    TEST_ASSERT( domain.isLocalState(i) );
	}
	else
	{
	    TEST_ASSERT( !domain.isLocalState(i) );
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardDomain, SomeOverlap )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> first_columns( 1, 0 );
    Teuchos::Array<double> first_values( 1, 2.0 );
    A->InsertGlobalValues( 0, first_columns().size(), 
                           &first_values[0], &first_columns[0] );
    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 2;
	values[1] = 3;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-1;
    A->InsertGlobalValues( global_num_rows-1, 1,
                           &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::ForwardDomain<VectorType,MatrixType> domain( B, x, plist );

    // Check the tally.
    Teuchos::RCP<VectorType> y = 
        VT::createFromRows( comm, domain.localStates()() );
    double y_val = 5.0;
    VT::putScalar( *y, y_val );
    double x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setSource( y );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, x_val );
	    history.live();
	    tally->tallyHistory( history );
            history.setEvent( MCLS::Event::CUTOFF );
            history.kill();
            tally->postProcessHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const double> x_view = VT::view( *x );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	if ( comm_rank == 0 || i > 1 )
	{
	    TEST_EQUALITY( x_view[i], x_val*y_val );
	}
	else
	{
	    TEST_EQUALITY( x_view[i], 2*x_val*y_val );
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
		TEST_ASSERT( domain.isLocalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isLocalState(i) );
	    }
	}
    }
    else
    {
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	    {
		TEST_ASSERT( domain.isLocalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isLocalState(i) );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardDomain, SomeOverlapPackUnpack )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> first_columns( 1, 0 );
    Teuchos::Array<double> first_values( 1, 2.0 );
    A->InsertGlobalValues( 0, first_columns().size(), 
                           &first_values[0], &first_columns[0] );
    Teuchos::Array<int> global_columns( 2 );
    Teuchos::Array<double> values( 2 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i;
	global_columns[1] = i+1;
	values[0] = 2;
	values[1] = 3;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-1;  
    A->InsertGlobalValues( global_num_rows-1, 1,
                           &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::ForwardDomain<VectorType,MatrixType> primary_domain( B, x, plist );

    // Pack the domain into a buffer.
    Teuchos::Array<char> domain_buffer = primary_domain.pack();

    // Unpack the domain to make a new one for testing.
    MCLS::ForwardDomain<VectorType,MatrixType> domain( domain_buffer, comm );

    // Check the tally.
    Teuchos::RCP<VectorType> y = 
        VT::createFromRows( comm, domain.localStates()() );
    double y_val = 5.0;
    VT::putScalar( *y, y_val );
    double x_val = 2;
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setBaseVector( x );
    tally->setSource( y );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, x_val );
	    history.live();
	    tally->tallyHistory( history );
            history.setEvent( MCLS::Event::CUTOFF );
            history.kill();
            tally->postProcessHistory( history );
	}
    }

    tally->combineSetTallies( comm );

    Teuchos::ArrayRCP<const double> x_view = VT::view( *x );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	if ( comm_rank == 0 || i > 1 )
	{
	    TEST_EQUALITY( x_view[i], x_val*y_val );
	}
	else
	{
	    TEST_EQUALITY( x_view[i], 2*x_val*y_val );
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
		TEST_ASSERT( domain.isLocalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isLocalState(i) );
	    }
	}
    }
    else
    {
	for ( int i = 0; i < global_num_rows; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	    {
		TEST_ASSERT( domain.isLocalState(i) );
	    }
	    else
	    {
		TEST_ASSERT( !domain.isLocalState(i) );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardDomain, Transition )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1 );
    for ( int i = 1; i < global_num_rows; ++i )
    {
	global_columns[0] = i-1;
	values[0] = -0.5;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    values[0] = -0.5;
    global_columns[0] = global_num_rows-1;
    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = MT::copyTranspose(*A);
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::ForwardDomain<VectorType,MatrixType> domain( B, x, plist );

    // Set the source with the tally.
    Teuchos::RCP<VectorType> y = 
        VT::createFromRows( comm, domain.localStates()() );
    double y_val = 5.0;
    VT::putScalar( *y, y_val );
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setSource( y );

    // Process a history transition in the domain.
    MCLS::RNGControl control( 2394723 );
    MCLS::RNGControl::RNG rng = control.rng( 4 );
    double weight = 3.0; 
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	if ( comm_rank == comm_size - 1 )
	{
	    if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, weight );
		history.live();
		history.setEvent( MCLS::Event::TRANSITION );
		history.setRNG( rng );
		domain.processTransition( history );

		TEST_EQUALITY( history.state(), i+1 );
		TEST_EQUALITY( history.weight(), weight/2 );
	    }
	}
	else
	{
	    if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, weight );
		history.live();
		history.setEvent( MCLS::Event::TRANSITION );
		history.setRNG( rng );
		domain.processTransition( history );

		TEST_EQUALITY( history.state(), i+1 );
		TEST_EQUALITY( history.weight(), weight/2 );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardDomain, Diagonal )
{
    typedef Epetra_Vector VectorType; 
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef MCLS::ForwardTally<VectorType> TallyType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear operator and solution vector.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 1 );
    Teuchos::Array<double> values( 1 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	global_columns[0] = i;
	values[0] = 3.0;
	A->InsertGlobalValues( i, global_columns().size(), 
			       &values[0], &global_columns[0] );
    }
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );

    // Build the forward domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    MCLS::ForwardDomain<VectorType,MatrixType> domain( B, x, plist );

    // Set the source with the tally.
    Teuchos::RCP<VectorType> y = 
        VT::createFromRows( comm, domain.localStates()() );
    double y_val = 5.0;
    VT::putScalar( *y, y_val );
    Teuchos::RCP<TallyType> tally = domain.domainTally();
    tally->setSource( y );

    // Process a history transition in the domain.
    MCLS::RNGControl control( 2394723 );
    MCLS::RNGControl::RNG rng = control.rng( 4 );
    double weight = 3.0; 
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, weight );
	    history.live();
	    history.setEvent( MCLS::Event::TRANSITION );
	    history.setRNG( rng );
	    domain.processTransition( history );

	    TEST_EQUALITY( history.state(), i );
	    TEST_EQUALITY( history.weight(), -weight*2.0 );
	}
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraForwardDomain.cpp
//---------------------------------------------------------------------------//

