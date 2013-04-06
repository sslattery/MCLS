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

#include <MCLS_DomainTransporter.hpp>
#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>
#include <MCLS_History.hpp>
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
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( DomainTransporter, Typedefs, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::History<GO> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

    typedef MCLS::DomainTransporter<DomainType> TransportType;
    typedef typename TransportType::HistoryType history_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
}

UNIT_TEST_INSTANTIATION( DomainTransporter, Typedefs )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( DomainTransporter, Cutoff, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<GO> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

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
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A, x, plist ) );

    // Build the domain transporter.
    double weight = 3.0; 
    double relative_cutoff = weight / 2 + 0.01;
    MCLS::DomainTransporter<DomainType> transporter( domain, plist );
    transporter.setCutoff( relative_cutoff );

    // Transport histories through the domain.
    MCLS::RNGControl control( 2394723 );
    MCLS::RNGControl::RNG rng = control.rng( 4 );
    for ( int i = 0; i < global_num_rows-1; ++i )
    {
	if ( comm_rank == comm_size - 1 )
	{
	    if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, weight );
		history.live();
		history.setRNG( rng );
		transporter.transport( history );

		TEST_EQUALITY( history.state(), i+1 );
		TEST_EQUALITY( history.weight(), weight / 2 );
		TEST_EQUALITY( history.event(), MCLS::CUTOFF );
		TEST_ASSERT( !history.alive() );
	    }
	}
	else
	{
	    if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, weight );
		history.live();
		history.setRNG( rng );
		transporter.transport( history );

		TEST_EQUALITY( history.state(), i+1 );
		TEST_EQUALITY( history.weight(), weight / 2 );
		TEST_EQUALITY( history.event(), MCLS::CUTOFF );
		TEST_ASSERT( !history.alive() );
	    }
	}
    }

    // Check the tally.
    domain->domainTally()->combineSetTallies();
    Teuchos::ArrayRCP<const Scalar> x_view = 
        VT::view( *x );
    Scalar x_val = weight;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	if ( comm_rank == comm_size-1 && i == local_num_rows-1 )
	{
	    TEST_EQUALITY( x_view[i], 0 );
	}
	else if ( comm_rank == 0 || i > 1 )
	{
	    TEST_EQUALITY( x_view[i], x_val );
	}
	else
	{
	    TEST_EQUALITY( x_view[i], 2*x_val );
	}
    }
}

UNIT_TEST_INSTANTIATION( DomainTransporter, Cutoff )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( DomainTransporter, Cutoff2, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<GO> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

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
	values[0] = -0.5/comm_size;
	A->insertGlobalValues( i, global_columns(), values() );
    }
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5/comm_size;
    A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
    A->fillComplete();

    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A, x, plist ) );

    // Build the domain transporter.
    double weight = 3.0; 
    double relative_cutoff = weight / 4 + 0.01;
    MCLS::DomainTransporter<DomainType> transporter( domain, plist );
    transporter.setCutoff( relative_cutoff );

    // Transport histories through the domain.
    MCLS::RNGControl control( 2394723 );
    MCLS::RNGControl::RNG rng = control.rng( 4 );
    for ( int i = 0; i < global_num_rows-2; ++i )
    {
	if ( comm_rank == comm_size - 1 )
	{
	    if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, weight );
		history.live();
		history.setRNG( rng );
		transporter.transport( history );

		TEST_EQUALITY( history.state(), i+2 );
		TEST_EQUALITY( history.weight(), weight / 4 );
		TEST_EQUALITY( history.event(), MCLS::CUTOFF );
		TEST_ASSERT( !history.alive() );
	    }
	}
	else
	{
	    if ( i >= local_num_rows*comm_rank && i < 1+local_num_rows*(comm_rank+1) )
	    {
		HistoryType history( i, weight );
		history.live();
		history.setRNG( rng );
		transporter.transport( history );

		TEST_EQUALITY( history.state(), i+2 );
		TEST_EQUALITY( history.weight(), weight / 4 );
		TEST_EQUALITY( history.event(), MCLS::CUTOFF );
		TEST_ASSERT( !history.alive() );
	    }
	}
    }

    // Check the tally.
    domain->domainTally()->combineSetTallies();
    Teuchos::ArrayRCP<const Scalar> x_view = 
        VT::view( *x );
    Scalar x_val = weight;
    for ( int i = 0; i < local_num_rows; ++i )
    {
	if ( comm_rank == comm_size-1 && i == local_num_rows-1 )
	{
	    TEST_EQUALITY( x_view[i], 0.0 );
	}
	else if ( comm_rank == comm_size-1 && i == local_num_rows-2 )
	{
	    TEST_EQUALITY( x_view[i], 3.0/2.0 );
	}
	else if ( comm_rank == 0 && i == 0 )
	{
	    TEST_EQUALITY( x_view[i], x_val );
	}
	else if ( comm_rank == 0 || i > 2 )
	{
	    TEST_EQUALITY( x_view[i], 3.0*x_val/2.0 );
	}
	else
	{
	    TEST_EQUALITY( x_view[i], 3.0*x_val );
	}
    }
}

UNIT_TEST_INSTANTIATION( DomainTransporter, Cutoff2 )

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST_TEMPLATE_3_DECL( DomainTransporter, Boundary, LO, GO, Scalar )
{
    typedef Tpetra::Vector<Scalar,LO,GO> VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<GO> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

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
	Teuchos::RCP<const Tpetra::Map<LO,GO> > map = 
	    Tpetra::createUniformContigMap<LO,GO>( global_num_rows, comm );

	// Build the linear operator and solution vector.
	Teuchos::RCP<MatrixType> A = Tpetra::createCrsMatrix<Scalar,LO,GO>( map );
	Teuchos::Array<GO> global_columns( 3 );
	Teuchos::Array<Scalar> values( 3 );

	global_columns[0] = 0;
	global_columns[1] = 1;
	global_columns[2] = 2;
	values[0] = 2.2/comm_size;
	values[1] = 0.24/comm_size;
	values[2] = 0.24/comm_size;
	A->insertGlobalValues( 0, global_columns(), values() );
	for ( int i = 1; i < global_num_rows-1; ++i )
	{
	    global_columns[0] = i-1;
	    global_columns[1] = i;
	    global_columns[2] = i+1;
	    values[0] = 0.24/comm_size;
	    values[1] = 2.2/comm_size;
	    values[2] = 0.24/comm_size;
	    A->insertGlobalValues( i, global_columns(), values() );
	}
	global_columns[0] = global_num_rows-3;
	global_columns[1] = global_num_rows-2;
	global_columns[2] = global_num_rows-1;
	values[0] = 0.24/comm_size;
	values[1] = 0.24/comm_size;
	values[2] = 2.2/comm_size;
	A->insertGlobalValues( global_num_rows-1, global_columns(), values() );
	A->fillComplete();

	Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

	// Build the adjoint domain.
	Teuchos::ParameterList plist;
	plist.set<int>( "Overlap Size", 2 );
	Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( A, x, plist ) );

	// Build the domain transporter.
	MCLS::DomainTransporter<DomainType> transporter( domain, plist );
	transporter.setCutoff( 1.0e-12 );

	// Transport histories through the domain until they hit a boundary.
	double weight = 3.0; 
	MCLS::RNGControl control( 2394723 );
	MCLS::RNGControl::RNG rng = control.rng( 4 );
	for ( int i = 0; i < global_num_rows-1; ++i )
	{
	    if ( comm_rank == comm_size - 1 )
	    {
		if ( i >= local_num_rows*comm_rank && i < local_num_rows*(comm_rank+1) )
		{
		    HistoryType history( i, weight );
		    history.live();
		    history.setRNG( rng );
		    transporter.transport( history );

		    TEST_EQUALITY( history.event(), MCLS::BOUNDARY );
		    TEST_ASSERT( !history.alive() );
		}
	    }
	    else
	    {
		if ( i >= local_num_rows*comm_rank && i < 2+local_num_rows*(comm_rank+1) )
		{
		    HistoryType history( i, weight );
		    history.live();
		    history.setRNG( rng );
		    transporter.transport( history );

		    TEST_EQUALITY( history.event(), MCLS::BOUNDARY );
		    TEST_ASSERT( !history.alive() );
		}
	    }
	}
    }
}

UNIT_TEST_INSTANTIATION( DomainTransporter, Boundary )

//---------------------------------------------------------------------------//
// end tstTpetraDomainTransporter.cpp
//---------------------------------------------------------------------------//

