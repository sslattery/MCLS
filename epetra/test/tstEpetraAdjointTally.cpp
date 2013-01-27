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
 * \file tstTpetraAdjointTally.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra AdjointTally tests.
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

#include <MCLS_AdjointTally.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_History.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
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
TEUCHOS_UNIT_TEST( AdjointTally, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::History<int> HistoryType;
    typedef TallyType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AdjointTally, TallyHistory )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(inverse_rows.size()),
				inverse_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::AdjointTally<VectorType> tally( A, B );
    double a_val = 2;
    double b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }
    
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *a_view_iterator, a_val + b_val );
	}
	else
	{
	    TEST_EQUALITY( *a_view_iterator, a_val );
	}
    }

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator b_view_iterator;
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *b_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *b_view_iterator, b_val );
	}
    }

    TEST_EQUALITY( tally.numBaseRows(), VT::getLocalLength(*A) );
    Teuchos::Array<int> base_rows = tally.baseRows();
    TEST_EQUALITY( Teuchos::as<int>(base_rows.size()),
		   tally.numBaseRows() );
    for ( int i = 0; i < base_rows.size(); ++i )
    {
	TEST_EQUALITY( base_rows[i], VT::getGlobalRow(*A,i) )
    }

    TEST_EQUALITY( tally.numOverlapRows(), VT::getLocalLength(*B) );
    Teuchos::Array<int> overlap_rows = tally.overlapRows();
    TEST_EQUALITY( Teuchos::as<int>(overlap_rows.size()),
		   tally.numOverlapRows() );
    for ( int i = 0; i < overlap_rows.size(); ++i )
    {
	TEST_EQUALITY( overlap_rows[i], VT::getGlobalRow(*B,i) )
    }

    tally.zeroOut();
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	TEST_EQUALITY( *a_view_iterator, 0 );
    }
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	TEST_EQUALITY( *b_view_iterator, 0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AdjointTally, Combine )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(inverse_rows.size()),
				inverse_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::AdjointTally<VectorType> tally( A, B );

    double a_val = 2;
    double b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }

    tally.combineSetTallies();

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *a_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *a_view_iterator, a_val + b_val );
	}
    }

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator b_view_iterator;
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *b_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *b_view_iterator, b_val );
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( AdjointTally, Normalize )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::History<int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
    Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(inverse_rows.size()),
				inverse_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::AdjointTally<VectorType> tally( A, B );
    double a_val = 2;
    double b_val = 3;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, a_val );
	history.live();
	tally.tallyHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
    }
    
    tally.combineSetTallies();
    int nh = 10;
    tally.normalize( nh );

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *a_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *a_view_iterator, (a_val + b_val) / nh );
	}
    }

    Teuchos::ArrayRCP<const double> B_view = VT::view( *B );
    Teuchos::ArrayRCP<const double>::const_iterator b_view_iterator;
    for ( b_view_iterator = B_view.begin();
	  b_view_iterator != B_view.end();
	  ++b_view_iterator )
    {
	if ( comm_size == 1 )
	{
	    TEST_EQUALITY( *b_view_iterator, 0 );
	}
	else
	{
	    TEST_EQUALITY( *b_view_iterator, b_val );
	}
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraAdjointTally.cpp
//---------------------------------------------------------------------------//

