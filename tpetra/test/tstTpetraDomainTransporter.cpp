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
 * \file tstTpetraDomainTransporter.cpp
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

#include <MCLS_DomainTransporter.hpp>
#include <MCLS_AlmostOptimalDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_AdjointHistory.hpp>
#include <MCLS_AdjointTally.hpp>
#include <MCLS_Events.hpp>
#include <MCLS_PRNG.hpp>

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
// Test templates
//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainTransporter, Typedefs )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef Tpetra::CrsMatrix<double,int,long> MatrixType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AlmostOptimalDomain<VectorType,MatrixType,std::mt19937,TallyType> DomainType;
    typedef MCLS::AdjointHistory<long> HistoryType;

    typedef MCLS::DomainTransporter<DomainType> TransportType;
    typedef typename TransportType::HistoryType history_type;
    typedef typename TransportType::TallyType tally_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<TallyType, tally_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainTransporter, Cutoff )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<double,int,long> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<long> HistoryType;
    typedef std::mt19937 rng_type;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AlmostOptimalDomain<VectorType,MatrixType,rng_type,TallyType>
	DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<const Tpetra::Map<int,long> > map = 
	Tpetra::createUniformContigMap<int,long>( global_num_rows, comm );

    // Build the linear operator and solution vector.
    Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<double,int,long>( map );
    Teuchos::Array<long> global_columns( 1 );
    Teuchos::Array<double> values( 1, 0.5 );
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    global_columns[0] = i;
	    A->insertGlobalValues( i, global_columns(), values() );
	}
    }
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
    Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "History Length", 1 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A_T, x, plist ) );
    Teuchos::RCP<MCLS::PRNG<rng_type> > rng = Teuchos::rcp(
	new MCLS::PRNG<rng_type>( comm->getRank() ) );
    domain->setRNG( rng );

    // Build the domain transporter.
    double weight = 3.0; 
    MCLS::DomainTransporter<DomainType> transporter( domain );

    // Transport histories through the domain.
    for ( int i = 0; i < global_num_rows; ++i )
    {
	if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	{
	    HistoryType history( i, i, weight );
	    history.live();
	    transporter.transport( history );

	    TEST_EQUALITY( history.globalState(), i );
	    TEST_EQUALITY( history.localState(), i - local_num_rows*comm_rank );
	    TEST_EQUALITY( history.weight(), weight / 2 );
	    TEST_EQUALITY( history.event(), MCLS::Event::CUTOFF );
	    TEST_ASSERT( !history.alive() );
	}
    }

    // Check the tally.
    Teuchos::ArrayRCP<const double> x_view = VT::view( *x );
    double x_val = weight;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	TEST_EQUALITY( x_view[i], x_val );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainTransporter, Boundary )
{
    typedef Tpetra::Vector<double,int,long> VectorType;
    typedef Tpetra::CrsMatrix<double,int,long> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<long> HistoryType;
    typedef std::mt19937 rng_type;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AlmostOptimalDomain<VectorType,MatrixType,rng_type,TallyType>
	DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This test really needs a decomposed domain such that we can check
    // hitting the local domain boundary.
    if ( comm_size > 1 )
    {
	int local_num_rows = 10;
	int global_num_rows = local_num_rows*comm_size;
	Teuchos::RCP<const Tpetra::Map<int,long> > map = 
	    Tpetra::createUniformContigMap<int,long>( global_num_rows, comm );

	// Build the linear operator and solution vector.
	Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<double,int,long>( map );
	Teuchos::Array<long> global_columns( 3 );
	Teuchos::Array<double> values( 3 );

	global_columns[0] = 0;
	global_columns[1] = 1;
	global_columns[2] = 2;
	values[0] = 1.0/comm_size;
	values[1] = -0.499/comm_size;
	values[2] = -0.499/comm_size;
	A->insertGlobalValues( 0, global_columns(), values() );
	for ( int i = 1; i < global_num_rows-1; ++i )
	{
	    global_columns[0] = i-1;
	    global_columns[1] = i;
	    global_columns[2] = i+1;
	    values[0] = -0.499/comm_size;
	    values[1] = 1.0/comm_size;
	    values[2] = -0.499/comm_size;
	    A->insertGlobalValues( i, global_columns(), values() );
	}
	global_columns[0] = global_num_rows-3;
	global_columns[1] = global_num_rows-2;
	global_columns[2] = global_num_rows-1;
	values[0] = -0.499/comm_size;
	values[1] = -0.499/comm_size;
	values[2] = 1.0/comm_size;
	A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
	A->fillComplete();

	Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );
        Teuchos::RCP<MatrixType> A_T = MT::copyTranspose(*A);

	// Build the adjoint domain.
	Teuchos::ParameterList plist;
	plist.set<int>( "History Length", 10000 );
	Teuchos::RCP<DomainType> domain = 
            Teuchos::rcp( new DomainType( A_T, x, plist ) );
	Teuchos::RCP<MCLS::PRNG<rng_type> > rng = Teuchos::rcp(
	    new MCLS::PRNG<rng_type>( comm->getRank() ) );
	domain->setRNG( rng );

	// Build the domain transporter.
	MCLS::DomainTransporter<DomainType> transporter( domain );

	// Transport histories through the domain until they hit a boundary.
	double weight = 3.0; 
	for ( int i = 0; i < global_num_rows-1; ++i )
	{
	    if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, i, weight );
		history.live();
		transporter.transport( history );

		TEST_ASSERT( history.event() ==  MCLS::Event::BOUNDARY );
		TEST_ASSERT( !history.alive() );
	    }
	}
    }
}

//---------------------------------------------------------------------------//
// end tstTpetraDomainTransporter.cpp
//---------------------------------------------------------------------------//

