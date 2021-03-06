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
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardTally, Typedefs )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef MCLS::ForwardTally<VectorType> TallyType;
    typedef MCLS::ForwardHistory<long> HistoryType;
    typedef typename TallyType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( ForwardTally, TallyHistory )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<long> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<int,long> > map_a = 
	Tpetra::createUniformContigMap<int,long>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<double,int,long>( map_a );

    Teuchos::Array<long> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<long> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<long> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    typename Teuchos::Array<long>::iterator unique_it = 
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<const Tpetra::Map<int,long> > map_b = 
	Tpetra::createNonContigMap<int,long>( tally_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<double,int,long>( map_b );

    MCLS::ForwardTally<VectorType> tally( A );
    double a_val = 2;
    double b_val = 3;
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

    tally.finalize();

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
        TEST_EQUALITY( *a_view_iterator, a_val*b_val );
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
TEUCHOS_UNIT_TEST( ForwardTally, Normalize )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef MCLS::ForwardHistory<long> HistoryType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<int,long> > map_a = 
	Tpetra::createUniformContigMap<int,long>( global_num_rows, comm );
    Teuchos::RCP<VectorType> A = Tpetra::createVector<double,int,long>( map_a );

    Teuchos::Array<long> forward_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	forward_rows[i] = i + local_num_rows*comm_rank;
    }
    Teuchos::Array<long> inverse_rows( local_num_rows );
    for ( int i = 0; i < local_num_rows; ++i )
    {
	inverse_rows[i] = 
	    (local_num_rows-1-i) + local_num_rows*(comm_size-1-comm_rank);
    }
    Teuchos::Array<long> tally_rows( forward_rows.size() + inverse_rows.size() );
    std::sort( forward_rows.begin(), forward_rows.end() );
    std::sort( inverse_rows.begin(), inverse_rows.end() );
    std::merge( forward_rows.begin(), forward_rows.end(),
                inverse_rows.begin(), inverse_rows.end(),
                tally_rows.begin() );
    typename Teuchos::Array<long>::iterator unique_it =
        std::unique( tally_rows.begin(), tally_rows.end() );
    tally_rows.resize( std::distance(tally_rows.begin(),unique_it) );

    Teuchos::RCP<const Tpetra::Map<int,long> > map_b = 
	Tpetra::createNonContigMap<int,long>( tally_rows(), comm );
    Teuchos::RCP<VectorType> B = Tpetra::createVector<double,int,long>( map_b );

    MCLS::ForwardTally<VectorType> tally( A );

    double a_val = 2;
    double b_val = 3;
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
    
    tally.finalize();
    
    int nh = 10;
    tally.normalize( nh );

    Teuchos::ArrayRCP<const double> A_view = VT::view( *A );
    typename Teuchos::ArrayRCP<const double>::const_iterator a_view_iterator;
    for ( a_view_iterator = A_view.begin();
	  a_view_iterator != A_view.end();
	  ++a_view_iterator )
    {
        TEST_EQUALITY( *a_view_iterator, a_val*b_val );
    }
}

//---------------------------------------------------------------------------//
// end tstTpetraForwardTally.cpp
//---------------------------------------------------------------------------//

