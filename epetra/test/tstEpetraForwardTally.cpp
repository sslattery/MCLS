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
 * \file tstEpetraForwardTally.cpp
 * \author Stuart R. Slattery
 * \brief Epetra ForwardTally tests.
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

#include <MCLS_ForwardTally.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_ForwardHistory.hpp>
#include <MCLS_Events.hpp>

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
TEUCHOS_UNIT_TEST( ForwardTally, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::ForwardTally<VectorType> TallyType;
    typedef MCLS::ForwardHistory<int> HistoryType;
    typedef TallyType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardTally, TallyHistory )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<int> HistoryType;

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

    Teuchos::Array<int> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<int> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    Teuchos::Array<int>::iterator unique_it = 
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(tally_rows.size()),
				tally_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::ForwardTally<VectorType> tally( A, 0 );

    double a_val = 2.0;
    double b_val = 3.0;
    VT::putScalar( *B, b_val );
    tally.setSource( B );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, state, a_val );
	history.live();
	tally.tallyHistory( history );
	TEST_EQUALITY( history.historyTally(), a_val*b_val );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
	TEST_EQUALITY( history.historyTally(), b_val*b_val );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );
    }

    tally.combineSetTallies( comm );

    int histories_per_state = 2;
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
        TEST_EQUALITY( *a_view_iterator, 
		       (a_val*b_val+b_val*b_val) / histories_per_state );
    }

    TEST_EQUALITY( tally.numBaseRows(), VT::getLocalLength(*A) );
    Teuchos::Array<int> base_rows = tally.baseRows();
    TEST_EQUALITY( Teuchos::as<int>(base_rows.size()),
		   tally.numBaseRows() );
    for ( int i = 0; i < base_rows.size(); ++i )
    {
	TEST_EQUALITY( base_rows[i], VT::getGlobalRow(*A,i) )
    }

    tally.zeroOut();
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
	TEST_EQUALITY( *a_view_iterator, 0.0 );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardTally, SetCombine )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<int> HistoryType;

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

    Teuchos::Array<int> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<int> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    Teuchos::Array<int>::iterator unique_it = 
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(tally_rows.size()),
				tally_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::ForwardTally<VectorType> tally( A, 0 );

    // Sub in a map-compatible base vector to ensure we can swap vectors and
    // still do the parallel export operation.
    Teuchos::RCP<VectorType> C = VT::clone(*A);
    tally.setBaseVector( C );

    double a_val = 2;
    double b_val = 3;
    VT::putScalar( *B, b_val );
    tally.setSource( B );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, state, a_val );
	history.live();
	tally.tallyHistory( history );
	TEST_EQUALITY( history.historyTally(), a_val*b_val );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
	TEST_EQUALITY( history.historyTally(), b_val*b_val );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );
    }

    int histories_per_state = 2;
    Teuchos::ArrayRCP<const double> C_view = VT::view( *C );
    Teuchos::ArrayRCP<const double>::const_iterator c_view_iterator;
    for ( c_view_iterator = C_view.begin();
	  c_view_iterator != C_view.end();
	  ++c_view_iterator )
    {
        TEST_EQUALITY( *c_view_iterator, 0.0 );
    }

    tally.combineSetTallies( comm );

    for ( c_view_iterator = C_view.begin();
	  c_view_iterator != C_view.end();
	  ++c_view_iterator )
    {
        TEST_EQUALITY( *c_view_iterator, 
		       (a_val*b_val + b_val*b_val) / histories_per_state );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardTally, BlockCombine )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<int> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This test is for 4 procs.
    if ( comm_size == 4 )
    {
	// Build the set-constant communicator.
	Teuchos::Array<int> ranks(2);
	if ( comm_rank < 2 )
	{
	    ranks[0] = 0;
	    ranks[1] = 1;
	}
	else
	{
	    ranks[0] = 2;
	    ranks[1] = 3;
	}
	Teuchos::RCP<const Teuchos::Comm<int> > comm_set =
	    comm->createSubcommunicator( ranks() );
	int set_size = comm_set->getSize();
	int set_rank = comm_set->getRank();

	Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm_set );

	// Build the block-constant communicator.
	if ( comm_rank == 0 || comm_rank == 2 )
	{
	    ranks[0] = 0;
	    ranks[1] = 2;
	}
	else
	{
	    ranks[0] = 1;
	    ranks[1] = 3;
	}
	Teuchos::RCP<const Teuchos::Comm<int> > comm_block =
	    comm->createSubcommunicator( ranks() );
	int block_rank = comm_block->getRank();

	// Build the map.
	int local_num_rows = 10;
	int global_num_rows = local_num_rows*set_size;
	Teuchos::RCP<Epetra_Map> map_a = Teuchos::rcp(
	    new Epetra_Map( global_num_rows, 0, *epetra_comm ) );
	Teuchos::RCP<VectorType> A = Teuchos::rcp( new Epetra_Vector( *map_a ) );

        Teuchos::Array<int> forward_rows( local_num_rows );
        for ( int i = 0; i < local_num_rows; ++i )
        {
            forward_rows[i] = i + local_num_rows*set_rank;
        }
        Teuchos::Array<int> inverse_rows( local_num_rows );
        for ( int i = 0; i < local_num_rows; ++i )
        {
            inverse_rows[i] = 
                (local_num_rows-1-i) + local_num_rows*(set_size-1-set_rank);
        }
        Teuchos::Array<int> tally_rows( forward_rows.size() + inverse_rows.size() );
        std::sort( forward_rows.begin(), forward_rows.end() );
        std::sort( inverse_rows.begin(), inverse_rows.end() );
        std::merge( forward_rows.begin(), forward_rows.end(),
                    inverse_rows.begin(), inverse_rows.end(),
                    tally_rows.begin() );
        Teuchos::Array<int>::iterator unique_it = 
            std::unique( tally_rows.begin(), tally_rows.end() );
        tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

	Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
	    new Epetra_Map( -1, 
			    Teuchos::as<int>(tally_rows.size()),
			    tally_rows.getRawPtr(),
			    0,
			    *epetra_comm ) );
	Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

	MCLS::ForwardTally<VectorType> tally( A, 0 );

	// Sub in a base vector over just set 0 after we have made the tally.
	Teuchos::RCP<VectorType> C;
	if ( comm_rank < 2 )
	{
	    C = VT::clone(*A);
	    tally.setBaseVector( C );
	}
	comm->barrier();

	double a_val = 2;
	double b_val = 3;
	if ( block_rank == 1 )
	{
	    a_val = 4;
	    b_val = 6;
	}
	comm->barrier();

	VT::putScalar( *B, b_val );
	tally.setSource( B );
	for ( int i = 0; i < local_num_rows; ++i )
	{
	    int state = i + local_num_rows*set_rank;
	    HistoryType history( state, state, a_val );
	    history.live();
	    tally.tallyHistory( history );
	    history.kill();
	    history.setEvent( MCLS::Event::CUTOFF );
	    tally.postProcessHistory( history );

	    int inverse_state = 
		(local_num_rows-1-i) + local_num_rows*(set_size-1-set_rank);
	    history = HistoryType( inverse_state, state, a_val );
	    history.live();
	    tally.tallyHistory( history );
	    history.kill();
	    history.setEvent( MCLS::Event::CUTOFF );
	    tally.postProcessHistory( history );
	}

	tally.combineSetTallies( comm_set );
	tally.combineBlockTallies( comm_block, 2 );

	// The base tallies should be combined across the blocks. The sets
	// tallied over different vectors.
	if ( comm_rank < 2 )
	{
	    int histories_per_state = 2;
	    Teuchos::ArrayRCP<const double> C_view = VT::view( *C );
	    Teuchos::ArrayRCP<const double>::const_iterator c_view_iterator;
	    for ( c_view_iterator = C_view.begin();
		  c_view_iterator != C_view.end();
		  ++c_view_iterator )
	    {
		TEST_EQUALITY( *c_view_iterator, 
			       (2*3+4*6) / histories_per_state );
	    }
	}
	else
	{
	    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
	    Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
	    for ( a_view_iterator = A_view.begin();
		  a_view_iterator != A_view.end();
		  ++a_view_iterator )
	    {
		TEST_EQUALITY( *a_view_iterator, 2*3+4*6 );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardTally, Normalize )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<int> HistoryType;

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

    Teuchos::Array<int> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<int> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<int> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    Teuchos::Array<int>::iterator unique_it =
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<Epetra_Map> map_b = Teuchos::rcp(
		new Epetra_Map( -1, 
				Teuchos::as<int>(tally_rows.size()),
				tally_rows.getRawPtr(),
				0,
				*epetra_comm ) );
    Teuchos::RCP<VectorType> B = Teuchos::rcp( new Epetra_Vector( *map_b ) );

    MCLS::ForwardTally<VectorType> tally( A, 0 );
    double a_val = 2;
    double b_val = 3;
    VT::putScalar( *B, b_val );
    tally.setSource( B );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	int state = i + local_num_rows*comm_rank;
	HistoryType history( state, state, a_val );
	history.live();
	tally.tallyHistory( history );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );

	int inverse_state = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
	history = HistoryType( inverse_state, inverse_state, b_val );
	history.live();
	tally.tallyHistory( history );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );
    }
    
    tally.combineSetTallies( comm );
    int nh = 10;
    tally.normalize( nh );

    int histories_per_state = 2;
    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
        TEST_EQUALITY( *a_view_iterator, 
		       (a_val*b_val+b_val*b_val) / histories_per_state );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraForwardTally.cpp
//---------------------------------------------------------------------------//

