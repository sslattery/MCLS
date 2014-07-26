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
 * \file tstEpetraMCSASolverManager.cpp
 * \author Stuart R. Slattery
 * \brief Epetra Monte Carlo synthetic acceleration solver manager tests.
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

#include <MCLS_MCSASolverManager.hpp>
#include <MCLS_LinearProblem.hpp>
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
TEUCHOS_UNIT_TEST( MCSASolverManager, one_by_one_adjoint )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear system. 
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 3 );
    Teuchos::Array<double> values( 3 );
    global_columns[0] = 0;
    global_columns[1] = 1;
    global_columns[2] = 2;
    values[0] = 1.0;
    values[1] = 0.14;
    values[2] = 0.0;
    A->InsertGlobalValues( 0, global_columns.size(), 
			   &values[0], &global_columns[0] );
    for ( int i = 1; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = 0.14;
	values[1] = 1.0;
	values[2] = 0.14;
	A->InsertGlobalValues( i, global_columns.size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-3;
    global_columns[1] = global_num_rows-2;
    global_columns[2] = global_num_rows-1;
    values[0] = 0.0;
    values[1] = 0.14;
    values[2] = 1.0;
    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;

    // Build the LHS. Put a large positive number here to be sure we are
    // clear the vector before solving.
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *x, 0.0 );

    // Build the RHS with negative numbers. this gives us a negative
    // solution. 
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *b, -1.0 );

    // Solver parameters.
    Teuchos::RCP<Teuchos::ParameterList> plist = 
	Teuchos::rcp( new Teuchos::ParameterList() );
    double cutoff = 1.0e-8;
    plist->set<std::string>("MC Type", "Adjoint");
    plist->set<double>("Convergence Tolerance", 1.0e-8);
    plist->set<int>("Maximum Iterations", 10);
    plist->set<double>("Weight Cutoff", cutoff);
    plist->set<int>("MC Check Frequency", 50);
    plist->set<bool>("Reproducible MC Mode",true);
    plist->set<int>("Overlap Size", 2);
    plist->set<int>("Number of Sets", 1);
    plist->set<int>("Set Number of Histories", 100 );
    plist->set<std::string>("Transport Type", "Global" );

    // Create the linear problem.
    Teuchos::RCP<MCLS::LinearProblem<VectorType,MatrixType> > linear_problem =
	Teuchos::rcp( new MCLS::LinearProblem<VectorType,MatrixType>(
			  B, x, b ) );

    // Create the solver.
    MCLS::MCSASolverManager<VectorType,MatrixType> 
	solver_manager( linear_problem, comm, plist );

    // Solve the problem.
    bool converged_status = solver_manager.solve();

    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );

    // Check that we got a negative solution.
    Teuchos::ArrayRCP<const double> x_view = VT::view(*x);
    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }

    // Now solve the problem with a positive source.
    VT::putScalar( *b, 2.0 );
    VT::putScalar( *x, 0.0 );
    linear_problem->setLHS(x);
    converged_status = solver_manager.solve();
    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
    	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset the domain and solve again with a positive source.
    VT::putScalar( *x, 0.0 );
    linear_problem->setLHS(x);
    solver_manager.setProblem( linear_problem );
    converged_status = solver_manager.solve();
    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
    	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset both and solve with a negative source.
    VT::putScalar( *b, -2.0 );
    VT::putScalar( *x, 0.0 );
    linear_problem->setLHS(x);
    converged_status = solver_manager.solve();
    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
    	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MCSASolverManager, two_by_two_adjoint )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This is a 4 processor test.
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

	// Declare the linear problem in the global scope.
	Teuchos::RCP<MCLS::LinearProblem<VectorType,MatrixType> > linear_problem;

	// Build the linear system on set 0.
	if ( comm_rank < 2 )
	{
	    int local_num_rows = 10;
	    int global_num_rows = local_num_rows*set_size;
	    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm_set );
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
	    values[0] = 1.0;
	    values[1] = 0.14;
	    values[2] = 0.14;
	    A->InsertGlobalValues( 0, global_columns.size(), 
				   &values[0], &global_columns[0] );
	    for ( int i = 1; i < global_num_rows-1; ++i )
	    {
		global_columns[0] = i-1;
		global_columns[1] = i;
		global_columns[2] = i+1;
		values[0] = 0.14;
		values[1] = 1.0;
		values[2] = 0.14;
		A->InsertGlobalValues( i, global_columns.size(), 
				       &values[0], &global_columns[0] );
	    }
	    global_columns[0] = global_num_rows-3;
	    global_columns[1] = global_num_rows-2;
	    global_columns[2] = global_num_rows-1;
	    values[0] = 0.14;
	    values[1] = 0.14;
	    values[2] = 1.0;
	    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
				   &values[0], &global_columns[0] );
	    A->FillComplete();

	    Teuchos::RCP<MatrixType> B = A;

	    // Build the LHS. Put a large positive number here to be sure we are
	    // clear the vector before solving.
	    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );
	    VT::putScalar( *x, 0.0 );

	    // Build the RHS with negative numbers. this gives us a negative
	    // solution. 
	    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *B );
	    VT::putScalar( *b, -1.0 );

	    // Create the linear problem.
	    linear_problem = Teuchos::rcp( 
		new MCLS::LinearProblem<VectorType,MatrixType>(B, x, b) );
	}
	comm->barrier();

	// Solver parameters.
	Teuchos::RCP<Teuchos::ParameterList> plist = 
	    Teuchos::rcp( new Teuchos::ParameterList() );
	double cutoff = 1.0e-4;
	plist->set<std::string>("MC Type", "Adjoint");
	plist->set<double>("Convergence Tolerance", 1.0e-8);
	plist->set<int>("Maximum Iterations", 10);
	plist->set<double>("Weight Cutoff", cutoff);
	plist->set<int>("MC Check Frequency", 50);
	plist->set<bool>("Reproducible MC Mode",true);
	plist->set<int>("Overlap Size", 2);
	plist->set<int>("Number of Sets", 2);
	plist->set<int>("Set Number of Histories", 100 );
	plist->set<std::string>("Transport Type", "Global" );

	// Create the solver.
	MCLS::MCSASolverManager<VectorType,MatrixType> 
	    solver_manager( linear_problem, comm, plist );

	// Solve the problem.
	bool converged_status = solver_manager.solve();

	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );
	if ( comm_rank < 2 )
	{
	    // Check that we got a negative solution.
	    Teuchos::ArrayRCP<const double> x_view = 
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();

	// Now solve the problem with a positive source.
	if ( comm_rank < 2 )
	{
	    Teuchos::RCP<VectorType> b = 
		MT::cloneVectorFromMatrixRows( *linear_problem->getOperator() );
	    VT::putScalar( *b, 2.0 );
	    linear_problem->setRHS( b );
	    VT::putScalar( *linear_problem->getLHS(), 0.0 );
	}
	comm->barrier();

	converged_status = solver_manager.solve();

	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    Teuchos::ArrayRCP<const double> x_view = 
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();

	// Reset the domain and solve again with a positive source.
	if ( comm_rank < 2 )
	{
	    VT::putScalar( *linear_problem->getLHS(), 0.0 );
	}
	comm->barrier();
	solver_manager.setProblem( linear_problem );
	converged_status = solver_manager.solve();
	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    Teuchos::ArrayRCP<const double> x_view = 
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();

	// Reset both and solve with a negative source.
	if ( comm_rank < 2 )
	{
	    Teuchos::RCP<VectorType> b = 
		MT::cloneVectorFromMatrixRows( *linear_problem->getOperator() );
	    VT::putScalar( *b, -2.0 );
	    linear_problem->setRHS( b );
	    VT::putScalar( *linear_problem->getLHS(), 0.0 );
	}
	comm->barrier();

	converged_status = solver_manager.solve();
	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    Teuchos::ArrayRCP<const double> x_view =
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MCSASolverManager, one_by_one_forward )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );
    int comm_size = comm->getSize();

    int local_num_rows = 10;
    int global_num_rows = local_num_rows*comm_size;
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( global_num_rows, 0, *epetra_comm ) );

    // Build the linear system. 
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 3 );
    Teuchos::Array<double> values( 3 );
    global_columns[0] = 0;
    global_columns[1] = 1;
    global_columns[2] = 2;
    values[0] = 1.0;
    values[1] = 0.14;
    values[2] = 0.0;
    A->InsertGlobalValues( 0, global_columns.size(), 
			   &values[0], &global_columns[0] );
    for ( int i = 1; i < global_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = 0.14;
	values[1] = 1.0;
	values[2] = 0.14;
	A->InsertGlobalValues( i, global_columns.size(), 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = global_num_rows-3;
    global_columns[1] = global_num_rows-2;
    global_columns[2] = global_num_rows-1;
    values[0] = 0.0;
    values[1] = 0.14;
    values[2] = 1.0;
    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
			   &values[0], &global_columns[0] );
    A->FillComplete();

    Teuchos::RCP<MatrixType> B = A;

    // Build the LHS. Put a large positive number here to be sure we are
    // clear the vector before solving.
    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *x, 0.0 );

    // Build the RHS with negative numbers. this gives us a negative
    // solution. 
    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *B );
    VT::putScalar( *b, -1.0 );

    // Solver parameters.
    Teuchos::RCP<Teuchos::ParameterList> plist = 
	Teuchos::rcp( new Teuchos::ParameterList() );
    double cutoff = 1.0e-8;
    plist->set<std::string>("MC Type", "Forward");
    plist->set<double>("Convergence Tolerance", 1.0e-8);
    plist->set<int>("Maximum Iterations", 20);
    plist->set<double>("Weight Cutoff", cutoff);
    plist->set<int>("MC Check Frequency", 50);
    plist->set<bool>("Reproducible MC Mode",true);
    plist->set<int>("Overlap Size", 2);
    plist->set<int>("Number of Sets", 1);
    plist->set<int>("Set Number of Histories", 100 );
    plist->set<std::string>("Transport Type", "Global" );

    // Create the linear problem.
    Teuchos::RCP<MCLS::LinearProblem<VectorType,MatrixType> > linear_problem =
	Teuchos::rcp( new MCLS::LinearProblem<VectorType,MatrixType>(
			  B, x, b ) );

    // Create the solver.
    MCLS::MCSASolverManager<VectorType,MatrixType> 
	solver_manager( linear_problem, comm, plist );

    // Solve the problem.
    bool converged_status = solver_manager.solve();

    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );

    // Check that we got a negative solution.
    Teuchos::ArrayRCP<const double> x_view = VT::view(*x);
    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }

    // Now solve the problem with a positive source.
    VT::putScalar( *b, 2.0 );
    VT::putScalar( *x, 0.0 );
    linear_problem->setLHS(x);
    converged_status = solver_manager.solve();
    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
    	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset the domain and solve again with a positive source.
    VT::putScalar( *x, 0.0 );
    linear_problem->setLHS(x);
    solver_manager.setProblem( linear_problem );
    converged_status = solver_manager.solve();
    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
    	TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
    }

    // Reset both and solve with a negative source.
    VT::putScalar( *b, -2.0 );
    VT::putScalar( *x, 0.0 );
    linear_problem->setLHS(x);
    converged_status = solver_manager.solve();
    TEST_ASSERT( converged_status );
    TEST_ASSERT( solver_manager.getConvergedStatus() );
    TEST_ASSERT( solver_manager.getNumIters() < 10 );
    TEST_ASSERT( solver_manager.achievedTol() > 0.0 );
    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
    {
    	TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
    }
}

//---------------------------------------------------------------------------//
TEUCHOS_UNIT_TEST( MCSASolverManager, two_by_two_forward )
{
    typedef Epetra_Vector VectorType;
    typedef MCLS::VectorTraits<VectorType> VT;
    typedef Epetra_RowMatrix MatrixType;
    typedef MCLS::MatrixTraits<VectorType,MatrixType> MT;
 
    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();
    int comm_rank = comm->getRank();

    // This is a 4 processor test.
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

	// Declare the linear problem in the global scope.
	Teuchos::RCP<MCLS::LinearProblem<VectorType,MatrixType> > linear_problem;

	// Build the linear system on set 0.
	if ( comm_rank < 2 )
	{
	    int local_num_rows = 10;
	    int global_num_rows = local_num_rows*set_size;
	    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm_set );
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
	    values[0] = 1.0;
	    values[1] = 0.14;
	    values[2] = 0.14;
	    A->InsertGlobalValues( 0, global_columns.size(), 
				   &values[0], &global_columns[0] );
	    for ( int i = 1; i < global_num_rows-1; ++i )
	    {
		global_columns[0] = i-1;
		global_columns[1] = i;
		global_columns[2] = i+1;
		values[0] = 0.14;
		values[1] = 1.0;
		values[2] = 0.14;
		A->InsertGlobalValues( i, global_columns.size(), 
				       &values[0], &global_columns[0] );
	    }
	    global_columns[0] = global_num_rows-3;
	    global_columns[1] = global_num_rows-2;
	    global_columns[2] = global_num_rows-1;
	    values[0] = 0.14;
	    values[1] = 0.14;
	    values[2] = 1.0;
	    A->InsertGlobalValues( global_num_rows-1, global_columns.size(), 
				   &values[0], &global_columns[0] );
	    A->FillComplete();

	    Teuchos::RCP<MatrixType> B = A;

	    // Build the LHS. Put a large positive number here to be sure we are
	    // clear the vector before solving.
	    Teuchos::RCP<VectorType> x = MT::cloneVectorFromMatrixRows( *B );
	    VT::putScalar( *x, 0.0 );

	    // Build the RHS with negative numbers. this gives us a negative
	    // solution. 
	    Teuchos::RCP<VectorType> b = MT::cloneVectorFromMatrixRows( *B );
	    VT::putScalar( *b, -1.0 );

	    // Create the linear problem.
	    linear_problem = Teuchos::rcp( 
		new MCLS::LinearProblem<VectorType,MatrixType>(B, x, b) );
	}
	comm->barrier();

	// Solver parameters.
	Teuchos::RCP<Teuchos::ParameterList> plist = 
	    Teuchos::rcp( new Teuchos::ParameterList() );
	double cutoff = 1.0e-4;
	plist->set<std::string>("MC Type", "Forward");
	plist->set<double>("Convergence Tolerance", 1.0e-8);
	plist->set<int>("Maximum Iterations", 20);
	plist->set<double>("Weight Cutoff", cutoff);
	plist->set<int>("MC Check Frequency", 50);
	plist->set<bool>("Reproducible MC Mode",true);
	plist->set<int>("Overlap Size", 2);
	plist->set<int>("Number of Sets", 2);
	plist->set<int>("Set Number of Histories", 100 );
	plist->set<std::string>("Transport Type", "Global" );

	// Create the solver.
	MCLS::MCSASolverManager<VectorType,MatrixType> 
	    solver_manager( linear_problem, comm, plist );

	// Solve the problem.
	bool converged_status = solver_manager.solve();

	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    // Check that we got a negative solution.
	    Teuchos::ArrayRCP<const double> x_view = 
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();

	// Now solve the problem with a positive source.
	if ( comm_rank < 2 )
	{
	    Teuchos::RCP<VectorType> b = 
		MT::cloneVectorFromMatrixRows( *linear_problem->getOperator() );
	    VT::putScalar( *b, 2.0 );
	    linear_problem->setRHS( b );
	    VT::putScalar( *linear_problem->getLHS(), 0.0 );
	}
	comm->barrier();

	converged_status = solver_manager.solve();

	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    Teuchos::ArrayRCP<const double> x_view = 
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();

	// Reset the domain and solve again with a positive source.
	if ( comm_rank < 2 )
	{
	    VT::putScalar( *linear_problem->getLHS(), 0.0 );
	}
	comm->barrier();
	solver_manager.setProblem( linear_problem );
	converged_status = solver_manager.solve();
	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    Teuchos::ArrayRCP<const double> x_view = 
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it > Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();

	// Reset both and solve with a negative source.
	if ( comm_rank < 2 )
	{
	    Teuchos::RCP<VectorType> b = 
		MT::cloneVectorFromMatrixRows( *linear_problem->getOperator() );
	    VT::putScalar( *b, -2.0 );
	    linear_problem->setRHS( b );
	    VT::putScalar( *linear_problem->getLHS(), 0.0 );
	}
	comm->barrier();

	converged_status = solver_manager.solve();
	TEST_ASSERT( converged_status );
	TEST_ASSERT( solver_manager.getConvergedStatus() );
	TEST_ASSERT( solver_manager.getNumIters() < 10 );

	if ( comm_rank < 2 )
	{
	    Teuchos::ArrayRCP<const double> x_view =
		VT::view( *linear_problem->getLHS() );
	    Teuchos::ArrayRCP<const double>::const_iterator x_view_it;
	    for ( x_view_it = x_view.begin(); x_view_it != x_view.end(); ++x_view_it )
	    {
		TEST_ASSERT( *x_view_it < Teuchos::ScalarTraits<double>::zero() );
	    }
	}
	comm->barrier();
    }
}

//---------------------------------------------------------------------------//
// end tstEpetraMCSASolverManager.cpp
//---------------------------------------------------------------------------//

