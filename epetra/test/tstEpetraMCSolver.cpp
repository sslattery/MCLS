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
 * \file tstEpetraMCSolver.cpp
 * \author Stuart R. Slattery
 * \brief Epetra Monte Carlo set solver tests.
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

#include <MCLS_MCSolver.hpp>
#include <MCLS_UniformAdjointSource.hpp>
#include <MCLS_AdjointDomain.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_MatrixTraits.hpp>
#include <MCLS_EpetraAdapter.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>

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
TEUCHOS_UNIT_TEST( MCSolver, solve )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
    typedef MCLS::History<int> HistoryType;
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
    values[0] = 0.24/comm_size;
    values[1] = 0.24/comm_size;
    values[2] = 1.0/comm_size;
    A->InsertGlobalValues( 0, global_columns.size(), 
			   &values[0], &global_columns[0] );
    for ( int i = 1; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = 0.24/comm_size;
	values[1] = 1.0/comm_size;
	values[2] = 0.24/comm_size;
	A->InsertGlobalValues( i, global_columns.size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-3;
    global_columns[1] = global_num_rows-2;
    global_columns[2] = global_num_rows-1;
    values[0] = 0.24/comm_size;
    values[1] = 0.24/comm_size;
    values[2] = 1.0/comm_size;
    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;

    // Build the LHS. Put a large positive number here to be sure we are
    // clear the vector before solving.
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *x, 100.0 );

    // Build the RHS with negative numbers. this gives us a negative
    // solution. 
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *b, -1.0 );

    // Create the solver.
    Teuchos::RCP<Teuchos::ParameterList> plist = 
	Teuchos::rcp( new Teuchos::ParameterList() );
    double cutoff = 1.0e-6;
    plist->set<double>("Weight Cutoff", cutoff);
    plist->set<int>("MC Check Frequency", 10);
    plist->set<bool>("Reproducible MC Mode",true);
    MCLS::MCSolver<SourceType> solver( comm, plist );

    // Build the adjoint domain.
    plist->set<int>( "Overlap Size", 2 );
    Teuchos::RCP<DomainType> domain = Teuchos::rcp( new DomainType( B, x, *plist ) );

    // Create the adjoint source with a set number of histories.
    int mult = 10;
    plist->set<int>("Set Number of Histories", mult*global_num_rows);
    Teuchos::RCP<SourceType> source = Teuchos::rcp(
	new SourceType( b, domain, solver.rngControl(), comm, 
			comm->getSize(), comm->getRank(), *plist ) );
    source->buildSource();
    plist->set<double>("Relative Weight Cutoff", source->sourceWeight()*cutoff);

    // Set the Domain.
    solver.setDomain( domain );

    // Set the Source.
    solver.setSource( source );

    // Do solve the problem.
    solver.solve();

    // Check that we got a negative solution.
    Teuchos::ArrayRCP<const double> x_view = VT::view(*x);
    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }

    // Now solve the problem with a positive source.
    VT::putScalar( *b, 2.0 );
    source->buildSource();
    solver.setSource( source );
    plist->set<double>("Relative Weight Cutoff", source->sourceWeight()*cutoff);
    solver.solve();
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset the domain and solve again with a positive source.
    solver.setDomain( domain );
    solver.setSource( source );
    solver.solve();
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset both and solve with a negative source.
    VT::putScalar( *b, -2.0 );
    source->buildSource();
    plist->set<double>("Relative Weight Cutoff", source->sourceWeight()*cutoff*2);
    solver.setDomain( domain );
    solver.setSource( source );
    solver.solve();
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraMCSolver.cpp
//---------------------------------------------------------------------------//

