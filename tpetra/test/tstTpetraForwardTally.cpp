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
 * \file tstTpetraForwardTally.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra ForwardTally tests.
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
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_ForwardHistory.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>

#include <Tpetra_Map.hpp>
#include <Tpetra_Vector.hpp>

//---------------------------------------------------------------------------//
// Instantiation macro. 
// 
// These types are those enabled by Tpetra under explicit instantiation.
//---------------------------------------------------------------------------//
#define UNIT_TEST_INSTANTIATION( type, name )			           \
    TEUCHOS_UNIT_TEST_TEMPLATE_3_INSTANT( type, name, int, long, double )

//---------------------------------------------------------------------------//
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( ForwardTally, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::ForwardTally<VectorType> TallyType;
    typedef MCLS::ForwardHistory<GO> HistoryType;
    typedef typename TallyType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( ForwardTally, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( ForwardTally, TallyHistory, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<GO> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

    Teuchos::Array<GO> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<GO> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    typename Teuchos::Array<GO>::iterator unique_it = 
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( tally_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

    MCLS::ForwardTally<VectorType> tally( A, 0 );
    Scalar a_val = 2;
    Scalar b_val = 3;
    VT::putScalar( *B, b_val );
    tally.setSource( B );
    for ( int i = 0; i < tally_rows.size(); ++i )
    {
	HistoryType history( tally_rows[i], i, a_val );
	history.live();
	tally.tallyHistory( history );
	TEST_EQUALITY( history.historyTally(), a_val*b_val );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );
    }

    tally.combineSetTallies( comm );

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
        TEST_EQUALITY( *a_view_iterator, a_val*b_val );
    }

    TEST_EQUALITY( tally.numBaseRows(), VT::getLocalLength(*A) );
    Teuchos::Array<GO> base_rows = tally.baseRows();
    TEST_EQUALITY( Teuchos::as<GO>(base_rows.size()),
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

UNIT_TEST_INSTANTIATION( ForwardTally, TallyHistory )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( ForwardTally, SetCombine, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<GO> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

    Teuchos::Array<GO> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<GO> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    typename Teuchos::Array<GO>::iterator unique_it = 
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( tally_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

    MCLS::ForwardTally<VectorType> tally( A, 0 );

    // Sub in a map-compatible base vector to ensure we can swap vectors and
    // still do the parallel export operation.
    Teuchos::RCP<VectorType> C = VT::clone(*A);
    tally.setBaseVector( C );

    // Do the tallies.
    Scalar a_val = 2;
    Scalar b_val = 3;
    VT::putScalar( *B, b_val );
    tally.setSource( B );
    for ( int i = 0; i < tally_rows.size(); ++i )
    {
	HistoryType history( tally_rows[i], i, a_val );
	history.live();
	tally.tallyHistory( history );
	TEST_EQUALITY( history.historyTally(), a_val*b_val );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );
    }
    
    Teuchos::ArrayRCP<const Scalar> C_view = VT::view( *C );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator c_view_iterator;
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
        TEST_EQUALITY( *c_view_iterator, a_val*b_val );
    }
}

UNIT_TEST_INSTANTIATION( ForwardTally, SetCombine )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( ForwardTally, BlockCombine, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<GO> HistoryType;

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
	Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	    Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm_set );
	Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

        Teuchos::Array<GO> forward_rows( local_num_rows );
        for ( int i = 0; i < local_num_rows; ++i )
        {
            forward_rows[i] = i + local_num_rows*set_rank;
        }
        Teuchos::Array<GO> inverse_rows( local_num_rows );
        for ( int i = 0; i < local_num_rows; ++i )
        {
            inverse_rows[i] = 
                (local_num_rows-1-i) + local_num_rows*(set_size-1-set_rank);
        }
        Teuchos::Array<GO> tally_rows( forward_rows.size() + inverse_rows.size() );
        std::sort( forward_rows.begin(), forward_rows.end() );
        std::sort( inverse_rows.begin(), inverse_rows.end() );
        std::merge( forward_rows.begin(), forward_rows.end(),
                    inverse_rows.begin(), inverse_rows.end(),
                    tally_rows.begin() );
        typename Teuchos::Array<GO>::iterator unique_it = 
            std::unique( tally_rows.begin(), tally_rows.end() );
        tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

        Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
            Tpetra::createNonContigMap<LO,GO>( tally_rows(), comm_set );
        Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

        MCLS::ForwardTally<VectorType> tally( A, 0 );

	// Sub in a base vector over just set 0 after we have made the tally.
	Teuchos::RCP<VectorType> C;
	if ( comm_rank < 2 )
	{
	    C = VT::clone(*A);
	    tally.setBaseVector( C );
	}
	comm->barrier();

	Scalar a_val = 2;
	Scalar b_val = 3;
	if ( block_rank == 1 )
	{
	    a_val = 4;
	    b_val = 6;
	}
	comm->barrier();

	VT::putScalar( *B, b_val );
	tally.setSource( B );
	for ( int i = 0; i < tally_rows.size(); ++i )
	{
	    HistoryType history( tally_rows[i], i, a_val );
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
	    Teuchos::ArrayRCP<const Scalar> C_view = VT::view( *C );
	    typename Teuchos::ArrayRCP<const Scalar>::const_iterator c_view_iterator;
	    for ( c_view_iterator = C_view.begin();
		  c_view_iterator != C_view.end();
		  ++c_view_iterator )
	    {
		TEST_EQUALITY( *c_view_iterator, 2.0+3.0+4.0+6.0 );
	    }
	}
	else
	{
	    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
	    typename Teuchos::ArrayRCP<const Scalar>::const_iterator a_view_iterator;
	    for ( a_view_iterator = A_view.begin();
		  a_view_iterator != A_view.end();
		  ++a_view_iterator )
	    {
		TEST_EQUALITY( *a_view_iterator, 0.0 );
	    }
	}
    }
}

UNIT_TEST_INSTANTIATION( ForwardTally, BlockCombine )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( ForwardTally, Normalize, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<GO> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_a = 
	Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<Scalar,LO,GO>( map_a );

    Teuchos::Array<GO> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<GO> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<GO> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    typename Teuchos::Array<GO>::iterator unique_it =
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<const Tpetra::Map<LO,GO> > map_b = 
	Tpetra::createNonContigMap<LO,GO>( tally_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<Scalar,LO,GO>( map_b );

    MCLS::ForwardTally<VectorType> tally( A, 0 );

    Scalar a_val = 2;
    Scalar b_val = 3;
    VT::putScalar( *B, b_val );
    tally.setSource( B );
    for ( int i = 0; i < tally_rows.size(); ++i )
    {
	HistoryType history( tally_rows[i], i, a_val );
	history.live();
	tally.tallyHistory( history );
	history.kill();
	history.setEvent( MCLS::Event::CUTOFF );
	tally.postProcessHistory( history );
    }
    
    tally.combineSetTallies( comm );
    int nh = 10;
    tally.normalize( nh );

    Teuchos::ArrayRCP<const Scalar> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const Scalar>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
        TEST_EQUALITY( *a_view_iterator, a_val*b_val );
    }
}

UNIT_TEST_INSTANTIATION( ForwardTally, Normalize )

//---------------------------------------------------------------------------//
// end tstTpetraForwardTally.cpp
//---------------------------------------------------------------------------//

