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
 * \file tstEpetraSourceTransportercpp
 * \author Stuart R. Slattery
 * \brief Epetra AdjointDomain tests.
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

#include <MCLS_SourceTransporter.hpp>
#include <MCLS_UniformAdjointSource.hpp>
#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>
#include <MCLS_AdjointHistory.hpp>
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
#include <Teuchos_DefaultMpiComm.hpp>
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
TEUCHOS_UNIT_TEST( SourceTransporter, Typedefs )
{
    typedef Epetra_Vector VectorType;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::AdjointHistory<int> HistoryType;
    typedef MCLS::UniformAdjointSource<DomainType> SourceType;
    typedef std::stack<Teuchos::RCP<HistoryType> > BankType;

    typedef MCLS::SourceTransporter<SourceType> SourceTransporterType;
    typedef SourceTransporterType::HistoryType history_type;
    typedef SourceTransporterType::BankType bank_type;
    typedef SourceTransporterType::Domain domain_type;

    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<HistoryType, history_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<BankType, bank_type>::value)
	== true, true );
    TEST_EQUALITY_CONST( 
	(Teuchos::TypeTraits::is_same<DomainType, domain_type>::value)
	== true, true );
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( SourceTransporter, transport )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::AdjointHistory<int> HistoryType;
    typedef MCLS::AdjointDomain<VectorType,MatrixType> DomainType;
    typedef MCLS::UniformAdjointSource<DomainType> SourceType;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear system. This operator is symmetric with a spectral
    // radius less than 1.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 3 );
    Teuchos::Array<double> values( 3 );
    global_columns[0] = 0;
    global_columns[1] = 1;
    global_columns[2] = 2;
    values[0] = 1.0/comm_size;
    values[1] = 0.12/comm_size;
    values[2] = 0.0/comm_size;
    A->InsertGlobalValues( 0, global_columns.size(), 
			   &values[0], &global_columns[0] );
    for ( int i = 1; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = 0.12/comm_size;
	values[1] = 1.0/comm_size;
	values[2] = 0.12/comm_size;
	A->InsertGlobalValues( i, global_columns.size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-3;
    global_columns[1] = global_num_rows-2;
    global_columns[2] = global_num_rows-1;
    values[0] = 0.0/comm_size;
    values[1] = 0.12/comm_size;
    values[2] = 1.0/comm_size;
    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = MT::copyTranspose(*A);
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *x, 0.0 );
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *b, -1.0 );

    // Build the adjoint domain.
    Teuchos::ParameterList plist;
    plist.set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( B, x, plist ) );

    // History setup.
    Teuchos::RCP<MCLS::RNGControl> control = Teuchos::rcp(
	new MCLS::RNGControl( 3939294 ) );
    HistoryType::setByteSize( control->getSize() );

    // Create the adjoint source with a set number of histories.
    int mult = 100;
    double cutoff = 1.0e-6;
    plist.set<int>("Set Number of Histories", mult*global_num_rows);
    plist.set<double>("Weight Cutoff", cutoff);
    Teuchos::RCP<SourceType> source = Teuchos::rcp(
	new SourceType( b, domain, control, comm, 
			comm->getSize(), comm->getRank(), plist ) );
    source->buildSource();

    // Create the source transporter.
    plist.set<int>("MC Check Frequency", 10);
    double relative_cutoff = source->sourceWeight()*cutoff;
    MCLS::SourceTransporter<SourceType> source_transporter( 
	comm, domain, plist );
    source_transporter.assignSource( source, relative_cutoff );

    // Do transport.
    source_transporter.transport();
    domain->domainTally()->combineSetTallies( comm );

    // Check that we got a negative solution.
    Teuchos::ArrayRCP<const double> x_view = 
        VT::view( *x );
    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < 0.0 );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraSourceTransportercpp
//---------------------------------------------------------------------------//

