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
 * \file tstEpetraDomainTransporter.cpp
 * \author Stuart R. Slattery
 * \brief Epetra AdjointDomain tests.
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
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_History.hpp>
#include <MCLS_AdjointTally.hpp>
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
TEUCHOS_UNIT_TEST( DomainTransporter, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

    typedef MCLS::DomainTransporter<DomainType> TransportType;
    typedef TransportType::HistoryType history_type;
    typedef TransportType::TallyType tally_type;

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
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

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
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5;
    A->InsertGlobalValues( global_num_rows-1, global_columns().size(),
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( B, x, plist ) );

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
    Teuchos::ArrayRCP<const double> x_view = 
        VT::view( domain->domainTally()->opDecompVector() );
    double x_val = weight;
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

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainTransporter, Cutoff2 )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

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
    global_columns[0] = global_num_rows-1;
    values[0] = -0.5;
    A->InsertGlobalValues( global_num_rows-1, global_columns.size(),
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( B, x, plist ) );

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
    Teuchos::ArrayRCP<const double> x_view = 
        VT::view( domain->domainTally()->opDecompVector() );
    double x_val = weight;
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

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( DomainTransporter, Boundary )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<int> HistoryType;
    typedef MCLS::AdjointTally<VectorType> TallyType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This test really needs a decomposed domain such that we can check
    // hitting the local domain boundary.
    if ( comm_size > 1 )
    {
	int local_num_rows = 10;
	int global_num_rows = local_num_rows*comm_size;
	Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	    new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

	// Build the linear operator and solution vector. This operator will
	// be assymetric so we quickly move the histories out of the domain
	// before they hit the low weight cutoff.
	Teuchos::RCP<Epetra_CrsMatrix> A = 	
	    Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
	Teuchos::Array<int> global_columns( 3 );
	Teuchos::Array<double> values( 3 );

	global_columns[0] = 0;
	global_columns[1] = 1;
	global_columns[2] = 2;
	values[0] = 1.0;
	values[1] = 0.49;
	values[2] = 0.49;
	A->InsertGlobalValues( 0, global_columns.size(),
			       &values[0], &global_columns[0] );
	for ( int i = 1; i < global_num_rows-1; ++i )
	{
	    global_columns[0] = i-1;
	    global_columns[1] = i;
	    global_columns[2] = i+1;
	    values[0] = 0.49;
	    values[1] = 1.0;
	    values[2] = 0.49;
	    A->InsertGlobalValues( i, global_columns.size(),
				   &values[0], &global_columns[0] );
	}
	global_columns[0] = global_num_rows-3;
	global_columns[1] = global_num_rows-2;
	global_columns[2] = global_num_rows-1;
	values[0] = 0.49;
	values[1] = 0.49;
	values[2] = 1.0;
	A->InsertGlobalValues( global_num_rows-1, global_columns().size(), 
			       &values[0], &global_columns[0] );
	A->FillComplete();

	Teuchos::RCP<MatrixType> B = A;
	Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *A );

	// Build the adjoint domain.
	Teuchos::ParameterList plist;
	plist.set<int>( "Overlap Size", 2 );
	Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( B, x, plist ) );

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

//---------------------------------------------------------------------------//
// end tstEpetraDomainTransporter.cpp
//---------------------------------------------------------------------------//

