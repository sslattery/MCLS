//---------------------------------------------------------------------------//
/*!
 * \file poisson_driver.cpp
 * \author Stuart R. Slattery
 * \brief 1D poisson driver.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_SolverFactory.hpp>
#include <MCLS_SolverManager.hpp>
#include <MCLS_LinearProblem.hpp>
#include <MCLS_EpetraAdapter.hpp>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_FancyOStream.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_OpaqueWrapper.hpp>
#include <Teuchos_TypeTraits.hpp>
#include <Teuchos_Tuple.hpp>
#include <Teuchos_ParameterList.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_XMLParameterListCoreHelpers.hpp"
#include <Teuchos_TimeMonitor.hpp>
#include <Teuchos_Time.hpp>

#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_RowMatrix.h>
#include <Epetra_Comm.h>
#include <Epetra_SerialComm.h>

#ifdef HAVE_MPI
#include <Epetra_MpiComm.h>
#include <Teuchos_DefaultMpiComm.hpp>
#endif

//---------------------------------------------------------------------------//
// Typedefs.
//---------------------------------------------------------------------------//

typedef Epetra_Vector Vector;
typedef MCLS::VectorTraits<Vector> VT;
typedef Epetra_RowMatrix Matrix;
typedef MCLS::MatrixTraits<Vector,Matrix> MT;

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
// Poisson operator.
Teuchos::RCP<Epetra_CrsMatrix> buildPoissonOperator(
    const int local_num_rows,
    const Teuchos::RCP<Epetra_Comm> epetra_comm )
{
    // Setup parallel distribution.
    Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(
	new Epetra_Map( local_num_rows, 0, *epetra_comm ) );

    // Build the linear operator.
    Teuchos::RCP<Epetra_CrsMatrix> A = 	
	Teuchos::rcp( new Epetra_CrsMatrix( Copy, *map, 0 ) );
    Teuchos::Array<int> global_columns( 3 );
    Teuchos::Array<double> values( 3 );
    global_columns[0] = 0;
    values[0] = 1.0;
    A->InsertGlobalValues( 0, 1, &values[0], &global_columns[0] );
    for ( int i = 1; i < local_num_rows-1; ++i )
    {
	global_columns[0] = i-1;
	global_columns[1] = i;
	global_columns[2] = i+1;
	values[0] = -0.5;
	values[1] = 1.0;
	values[2] = -0.5;
	A->InsertGlobalValues( i, 3, 
			       &values[0], &global_columns[0] );
    }
    global_columns[0] = local_num_rows-1;
    values[0] = 1.0;
    A->InsertGlobalValues( local_num_rows-1, 1, 
			   &values[0], &global_columns[0] );
    A->FillComplete();
    return A;
}

//---------------------------------------------------------------------------//
// Prolongation operator. 2h -> h. 
void applyProlongationOperator2( const Teuchos::ArrayView<const double>& v_2h,
				Teuchos::ArrayView<double>& v_h )
{
    assert( v_h.size() == 2 * v_2h.size() - 1 );

    for ( unsigned j = 0; j < v_2h.size() - 1; ++j )
    {
	v_h[2*j] = v_2h[j];
	v_h[2*j+1] = 0.5 * ( v_2h[j] + v_2h[j+1] );
    }

    v_h.back() = v_2h.back();
}

//---------------------------------------------------------------------------//
// Restriction operator. h -> 2h.
void applyRestrictionOperator2( const Teuchos::ArrayView<const double>& v_h,
			       Teuchos::ArrayView<double>& v_2h )
{
    assert( v_h.size() == 2 * v_2h.size() - 1 );

    v_2h.front() = v_h.front();

    for ( unsigned j = 1; j < v_2h.size() - 1; ++j )
    {
	v_2h[j] = 0.25 * ( v_h[2*j-1] + 2.0 * v_h[2*j] + v_h[2*j+1] );
    }

    v_2h.back() = v_h.back();
}

//---------------------------------------------------------------------------//
// Prolongation operator. 4h -> h. 
void applyProlongationOperator4( const Teuchos::ArrayView<const double>& v_4h,
				Teuchos::ArrayView<double>& v_h )
{
    assert( v_h.size() == 4*(v_4h.size()-1) + 1 );

    for ( unsigned j = 0; j < v_4h.size() - 1; ++j )
    {
	v_h[4*j] = v_4h[j];
	v_h[4*j+1] = 0.75*v_4h[j] + 0.25*v_4h[j+1];
	v_h[4*j+2] = 0.5*v_4h[j] + 0.5*v_4h[j+1];
	v_h[4*j+3] = 0.25*v_4h[j] + 0.75*v_4h[j+1];
    }

    v_h.back() = v_4h.back();
}

//---------------------------------------------------------------------------//
// Restriction operator. h -> 4h.
void applyRestrictionOperator4( const Teuchos::ArrayView<const double>& v_h,
			       Teuchos::ArrayView<double>& v_4h )
{
    assert( v_h.size() == 4*(v_4h.size()-1) + 1 );

    v_4h.front() = v_h.front();

    for ( unsigned j = 1; j < v_4h.size() - 1; ++j )
    {
	v_4h[j] = 0.1 * ( v_h[4*j-2] + 2*v_h[4*j-1] + 4*v_h[4*j] +
			    2*v_h[4*j+1] + v_h[4*j+2] );
    }

    v_4h.back() = v_h.back();
}

//---------------------------------------------------------------------------//
// Prolongation operator.
void applyProlongationOperator( const int M,
				const Teuchos::ArrayView<const double>& v_Mh,
				Teuchos::ArrayView<double>& v_h )
{
    if ( 2 == M )
    {
	applyProlongationOperator2( v_Mh, v_h );
    }

    else if ( 4 == M )
    {
	applyProlongationOperator4( v_Mh, v_h );
    }
}

//---------------------------------------------------------------------------//
// Restriction operator.
void applyRestrictionOperator( const int M,
			       const Teuchos::ArrayView<const double>& v_h,
			       Teuchos::ArrayView<double>& v_Mh )
{
    if ( 2 == M )
    {
	applyRestrictionOperator2( v_h, v_Mh );
    }

    else if ( 4 == M )
    {
	applyRestrictionOperator4( v_h, v_Mh );
    }
}

//---------------------------------------------------------------------------//
// Multilevel MC solve.
void multilevelSolve( Teuchos::RCP<Vector>& x,
		      const Teuchos::RCP<Vector>& rhs,
		      Teuchos::RCP<Teuchos::ParameterList>& plist,
		      const Teuchos::RCP<const Teuchos::Comm<int> >& comm )
{
    // Hierarchy parameters.
    int problem_size = plist->get<int>("Problem Size");
    int num_levels = plist->get<int>("Number of Levels");
    int set_histories = plist->get<int>("Set Number of Histories");
    int M = plist->get<int>("Grid Refinement");
    int cga = plist->get<int>("Coarse Grid Acceleration");

    // Create the number of histories hierarchy.
    Teuchos::Array<int> num_histories( num_levels );
    for ( int n = 0; n < num_levels; ++n )
    {
	num_histories[n] =
	    set_histories * std::pow( M, -3.0*(num_levels-n-1)/2.0 );
    }

    // Create the grid hierarchy.
    Teuchos::Array<int> grid_sizes( num_levels, problem_size );
    std::cout << std::endl << "Grid size 0: " << grid_sizes[0] << std::endl;
    for ( int n = 1; n < num_levels; ++n )
    {
	grid_sizes[n] = (grid_sizes[n-1] - 1) / M + 1;
	std::cout << "Grid size " << n << ": " << grid_sizes[n] << std::endl;
    }

    // Create the poisson operator hierarchy. These are scaled by the inverse
    // of the diagonal (2*h^2) already.
    Teuchos::Array<Teuchos::RCP<Matrix> > A(num_levels);
    for ( int n = 0; n < num_levels; ++n )
    {
	A[n] = buildPoissonOperator( grid_sizes[n], getEpetraComm(comm) );
    }

    // Create the RHS hierarchy. Scale by the inverse of the diagonal.
    Teuchos::Array<Teuchos::RCP<Epetra_Vector> > b(num_levels);
    b[0] = VT::deepCopy( *rhs );
    for ( int n = 1; n < num_levels; ++n )
    {
	b[n] = MT::cloneVectorFromMatrixRows( *A[n] );
    }
    for ( int i = 1; i < num_levels; ++i )
    {
	double* b_h_ptr;
	b[i-1]->ExtractView( &b_h_ptr );
	Teuchos::ArrayView<const double> b_h( b_h_ptr, grid_sizes[i-1] );
	double* b_Mh_ptr;
	b[i]->ExtractView( &b_Mh_ptr );
	Teuchos::ArrayView<double> b_Mh( b_Mh_ptr, grid_sizes[i] );
	applyRestrictionOperator( M, b_h, b_Mh );
    }
    
    // Scale the RHS hierarchy by the inverse of the diagonal at that level.
    double h = 0.0;
    for ( int i = 0; i < num_levels; ++i )
    {
    	h = 1.0 / (grid_sizes[i] - 1 );
    	for ( int j = 1; j < grid_sizes[i]-1; ++j )
    	{
    	    (*b[i])[j] *= 2.0*h*h;
    	}
    }

    // Create the solution vector hierarchy.
    Teuchos::Array<Teuchos::RCP<Epetra_Vector> > u(num_levels);
    for ( int n = 0; n < num_levels; ++n )
    {
	u[n] = MT::cloneVectorFromMatrixRows( *A[n] );
    }
    VT::putScalar( *u[0], 0.0 );
    for ( int i = 1; i < num_levels; ++i )
    {
    	double* u_h_ptr;
    	u[i-1]->ExtractView( &u_h_ptr );
    	Teuchos::ArrayView<const double> u_h( u_h_ptr, grid_sizes[i-1] );
    	double* u_Mh_ptr;
    	u[i]->ExtractView( &u_Mh_ptr );
    	Teuchos::ArrayView<double> u_Mh( u_Mh_ptr, grid_sizes[i] );
    	applyRestrictionOperator( M, u_h, u_Mh );
    }

    // Build the correction vector hierarchy.
    Teuchos::Array<Teuchos::RCP<Epetra_Vector> > d(num_levels);
    for ( int n = 0; n < num_levels; ++n )
    {
	d[n] = MT::cloneVectorFromMatrixRows( *A[n] );
	VT::putScalar( *d[n], 0.0 );
    }

    // Build the residual hierarchy.
    Teuchos::Array<Teuchos::RCP<Epetra_Vector> > r(num_levels);
    for ( int n = 0; n < num_levels; ++n )
    {
    	r[n] = MT::cloneVectorFromMatrixRows( *A[n] );
    	MT::apply( *A[n], *u[n], *r[n] );
    	VT::update( *r[n], -1.0, *b[n], 1.0 );
    }

    // Create the residual linear problem hierarchy.
    Teuchos::Array<Teuchos::RCP<MCLS::LinearProblem<Vector,Matrix> > >
	linear_problem( num_levels );
    for ( int n = 0; n < num_levels; ++n )
    {
	linear_problem[n] =
	    Teuchos::rcp( new MCLS::LinearProblem<Vector,Matrix>(
			      A[n], d[n], r[n] ) );
    }

    // Create the solver.
    MCLS::SolverFactory<Vector,Matrix> factory;
    Teuchos::RCP<MCLS::SolverManager<Vector,Matrix> > solver_manager =
	factory.create( "Adjoint MC", comm, plist );

    // Solve the problem hierarchy.
    std::cout << std::endl;
    Teuchos::Time timer("");
    timer.start(true);
    for ( int n = 0; n < num_levels; ++n )
    {
	std::cout << "Solving Level " << n << " with " 
		  << num_histories[n] << " samples..." << std::endl;
	plist->set<int>("Set Number of Histories", num_histories[n] );
	solver_manager->setParameters( plist );
	solver_manager->setProblem( linear_problem[n] );

	// If we aren't doing coarse grid acceleration, solve all levels.
	if ( !cga )
	{
	    solver_manager->solve();
	}
	else if ( n == num_levels - 1 )
	{
	    solver_manager->solve();
	}
    }
    timer.stop();
    std::cout << std::endl;

    // Apply the multilevel tally.
    for ( int n = 0; n < num_levels - 1; ++n )
    {
	// Get a view of the tally for this level.
	double* d_h_ptr;
	d[n]->ExtractView( &d_h_ptr );
	Teuchos::ArrayView<const double> d_h( d_h_ptr, grid_sizes[n] );

	// Restrict the tally to the coarse grid.
	Teuchos::Array<double> v_Mh( grid_sizes[n+1], 0.0 );
	Teuchos::ArrayView<double> v_Mh_view = v_Mh();
	applyRestrictionOperator( M, d_h, v_Mh_view );

	// Prolongate the tally back to the fine grid.
	Teuchos::Array<double> v_h( grid_sizes[n], 0.0 );
	Teuchos::ArrayView<double> v_h_view = v_h();
	applyProlongationOperator( M, v_Mh_view, v_h_view );

	// Subtract the coarse result from the fine result.
	for ( int i = 0; i < grid_sizes[n]; ++i )
	{
	    (*d[n])[i] -= v_h[i];
	}
    }

    // Collapse the tally hierarchy.
    for ( int n = num_levels - 1; n > 0; --n )
    {
	// Get a view of the tally for the coarse level.
	double* d_Mh_ptr;
	d[n]->ExtractView( &d_Mh_ptr );
	Teuchos::ArrayView<const double> d_Mh( d_Mh_ptr, grid_sizes[n] );

	// Prolongate the coarse level to the fine level.
	Teuchos::Array<double> v_h( grid_sizes[n-1], 0.0 );
	Teuchos::ArrayView<double> v_h_view = v_h();
	applyProlongationOperator( M, d_Mh, v_h_view );

	// Add the prolongated coarse result to the fine result.
	for ( int i = 0; i < grid_sizes[n-1]; ++i )
	{
	    // If we aren't doing coarse grid acceleration, sum all levels.
	    if ( !cga )
	    {
		(*d[n-1])[i] += v_h[i];
	    }
	    else
	    {
		(*d[n-1])[i] = v_h[i];
	    }
	}
    }

    // Apply the correction to the solution.
    VT::update( *x, 1.0, *d[0], 1.0 );
}

//---------------------------------------------------------------------------//
int main( int argc, char * argv[] )
{
    // Initialize parallel communication.
    Teuchos::GlobalMPISession mpi_session( &argc, &argv );
    Teuchos::RCP<const Teuchos::Comm<int> > comm =
	Teuchos::DefaultComm<int>::getComm();
    Teuchos::RCP<Epetra_Comm> epetra_comm = getEpetraComm( comm );

    // Initialize output stream.
    Teuchos::FancyOStream out( Teuchos::rcpFromRef( std::cout ) );
    out.setOutputToRootOnly( 0 );
    out.setShowProcRank( true );

    // Read in command line options.
    std::string xml_input_filename;
    Teuchos::CommandLineProcessor clp(false);
    clp.setOption( "xml-in-file",
		   &xml_input_filename,
		   "The XML file to read into a parameter list" );
    clp.parse(argc,argv);

    // Build the parameter list from the xml input.
    Teuchos::RCP<Teuchos::ParameterList> plist =
	Teuchos::rcp( new Teuchos::ParameterList() );
    Teuchos::updateParametersFromXmlFile(
	xml_input_filename, Teuchos::inoutArg(*plist) );

    // Hierarchy parameters.
    int problem_size = plist->get<int>("Problem Size");
    double h = 1.0 / (problem_size - 1 );

    // Create the poisson operator. This is scaled by the inverse
    // of the diagonal (2*h^2) already.
    Teuchos::RCP<Matrix> A = buildPoissonOperator( problem_size, epetra_comm );

    // Create the RHS. Scale by the inverse of the diagonal.
    Teuchos::RCP<Epetra_Vector> b = MT::cloneVectorFromMatrixRows( *A );
    double forcing = plist->get<double>("Forcing");
    VT::putScalar( *b, forcing );
    for ( int j = 1; j < problem_size-1; ++j )
    {
	(*b)[j] *= 2.0*h*h;
    }

    // Create the solution vector.
    Teuchos::RCP<Epetra_Vector> x = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *x, 0.0 );

    // Compute the initial residual.
    Teuchos::RCP<Epetra_Vector> r = MT::cloneVectorFromMatrixRows( *A );
    MT::apply( *A, *x, *r );
    VT::update( *r, -1.0, *b, 1.0 );
    Teuchos::RCP<Epetra_Vector> unscaled_r = MT::cloneVectorFromMatrixRows( *A );

    // Solve the problem using MCSA and multilevel mc.
    Teuchos::RCP<Epetra_Vector> d = MT::cloneVectorFromMatrixRows( *A );
    VT::putScalar( *d, 0.0 );
    double tol = plist->get<double>("Convergence Tolerance");
    double max_iters = plist->get<int>("MLMCSA Maximum Iterations");
    int num_iters = 0;
    double conv_check = 1.0;
    double r_norm = 0.0;
    double b_norm = 0.0;

    std::ofstream residual_file;
    residual_file.open( "residual.dat" );

    std::cout << std::endl;
    Teuchos::Time timer("");
    timer.start(true);
    while ( conv_check > tol && num_iters < max_iters )
    {
    	// Do a Richardson iteration.
    	VT::update( *x, 1.0, *r, 1.0 );

    	// Update the residual.
    	MT::apply( *A, *x, *r );
    	VT::update( *r, -1.0, *b, 1.0 );

	// Compute an unscaled residual.
	unscaled_r = VT::deepCopy( *r );
	for ( int j = 1; j < problem_size-1; ++j )
	{
	    (*unscaled_r)[j] *= 1.0 / (2.0*h*h);
	}

    	// Solve the multilevel Monte Carlo problem.
    	VT::putScalar( *d, 0.0 );
    	multilevelSolve( d, unscaled_r, plist, comm );
    	VT::update( *x, 1.0, *d, 1.0 );

    	// Update the residual.
    	MT::apply( *A, *x, *r );
    	VT::update( *r, -1.0, *b, 1.0 );

    	// Check for convergence and output.
	r_norm = VT::norm2( *r );
	b_norm = VT::norm2( *b );
    	conv_check = r_norm / b_norm;
    	std::cout << "MLMCSA Iteration : " << num_iters
    		  << ", Residual: " << conv_check << std::endl;
    	++num_iters;
	residual_file << std::setprecision(8) << conv_check << std::endl;
    }
    timer.stop();
    residual_file.close();
    std::cout << std::endl;

    // The point-wise error is the inf-norm of the top level residual.
    MT::apply( *A, *x, *r );
    VT::update( *r, -1.0, *b, 1.0 );
    double r_inf = VT::normInf( *r );
    std::cout << "||r||_inf: " << r_inf << std::endl;
    double r_2 = VT::norm2( *r );
    std::cout << "||r||_2: " << r_2 << std::endl;
    double toc = timer.totalElapsedTime();
    std::cout << std::endl << "Time: " << toc << std::endl;

    // Write the solution to a file.
    if ( comm->getRank() == 0 )
    {
        std::ofstream ofile;
        ofile.open( "solution.dat" );
        for ( int i = 0; i < problem_size; ++i )
        {
            ofile << std::setprecision(8) << (*x)[i] << std::endl;
        }
        ofile.close();
    }
    comm->barrier();

    // Compute the exact solution.
    MCLS::SolverFactory<Vector,Matrix> factory;
    Teuchos::RCP<MCLS::SolverManager<Vector,Matrix> > solver_manager =
	factory.create( "Fixed Point", comm, plist );
    Teuchos::RCP<Vector> x_exact = VT::clone( *x );
    Teuchos::RCP<MCLS::LinearProblem<Vector,Matrix> > exact_linear_problem =
	Teuchos::rcp( new MCLS::LinearProblem<Vector,Matrix>(
			  A, x_exact, b ) );
    solver_manager->setProblem( exact_linear_problem );
    solver_manager->solve();

    // Compute the error.
    double e_inf = 0.0;
    double e_2 = 0.0;
    double local_error = 0.0;
    for ( int i = 0; i < problem_size; ++i )
    {
	local_error = ( (*x_exact)[i] - (*x)[i] ) / (*x_exact)[i];
	e_inf = std::max( e_inf, std::abs(local_error) );
	e_2 += std::abs( local_error * local_error );
    }
    e_2 = std::sqrt(e_2);
    std::cout << "||e||_inf: " << e_inf << std::endl;
    std::cout << "||e||_2: " << e_2 << std::endl;
    std::cout << std::endl;

    // Write the exact solution to a file.
    if ( comm->getRank() == 0 )
    {
        std::ofstream ofile;
        ofile.open( "exact.dat" );
        for ( int i = 0; i < problem_size; ++i )
        {
            ofile << std::setprecision(8) << (*x_exact)[i] << std::endl;
        }
        ofile.close();
    }
    comm->barrier();

    return 0;
}

//---------------------------------------------------------------------------//
// end neutron_diffusion.cpp
//---------------------------------------------------------------------------//

