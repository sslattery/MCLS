//---------------------------------------------------------------------------//
/*!
 * \file neutron_diffusion.cpp
 * \author Stuart R. Slattery
 * \brief 2D Neutron diffusion driver.
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

#include "DiffusionProblem.hpp"
#include "Partitioner.hpp"

#include <MCLS_SolverFactory.hpp>
#include <MCLS_SolverManager.hpp>
#include <MCLS_LinearProblem.hpp>
#include <MCLS_TpetraAdapter.hpp>

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

#include <Tpetra_Map.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_Operator.hpp>

//---------------------------------------------------------------------------//
/* 
   2D neutron diffusion example.

   This example creates an operator for the 2D neutron diffusion problem with
   a 4th-order 9-point stencil for the laplacian and a uniform set of cross
   sections and sources across the domain.
*/

//---------------------------------------------------------------------------//
// Typedefs
//---------------------------------------------------------------------------//

typedef Tpetra::Vector<double,int,int> Vector;
typedef Tpetra::CrsMatrix<double,int,int> Matrix;

//---------------------------------------------------------------------------//
// Main
//---------------------------------------------------------------------------//
int main( int argc, char * argv[] )
{
    // Initialize parallel communication.
    Teuchos::GlobalMPISession mpi_session( &argc, &argv );
    Teuchos::RCP<const Teuchos::Comm<int> > comm =
	Teuchos::DefaultComm<int>::getComm();
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
    Teuchos::RCP<Teuchos::ParameterList> mcls_list = 
	Teuchos::rcpFromRef( plist->sublist("MCLS",true) );

    // Build a communicator for the primary set.
    int num_sets = mcls_list->get<int>("Number of Sets");
    int set_size = comm->getSize() / num_sets;
    int set_id = std::floor( Teuchos::as<double>(comm->getRank()) /
                             Teuchos::as<double>(set_size) );
    Teuchos::Array<int> comm_ranks( set_size );
    for ( int n = 0; n < set_size; ++n )
    {
        comm_ranks[n] = n;
    }
    Teuchos::RCP<const Teuchos::Comm<int> > set_comm = 
        comm->createSubcommunicator( comm_ranks() );

    // Setup the problem on the primary set.
    Teuchos::RCP<MCLS::LinearProblem<Vector,Matrix> > linear_problem;
    if ( 0 == set_id )
    {
	// Partition the problem.
	Teuchos::RCP<MCLSExamples::Partitioner> partitioner =
	    Teuchos::rcp( new MCLSExamples::Partitioner(set_comm, plist) );

	// Build operators and vectors.
	Teuchos::RCP<MCLSExamples::DiffusionProblem> diffusion_problem
	    = Teuchos::rcp(
		new MCLSExamples::DiffusionProblem(
		    set_comm, partitioner, plist, true) );

	// Extract the linear problem.
	linear_problem = Teuchos::rcp( 
	    new MCLS::LinearProblem<Vector,Matrix>(
		diffusion_problem->getOperator(),
		diffusion_problem->getLHS(),
		diffusion_problem->getRHS() ) );
    }
    comm->barrier();

    // Build the MCLS solver.
    std::string solver_type = plist->get<std::string>("Solver Type");
    MCLS::SolverFactory<Vector,Matrix> factory;
    Teuchos::RCP<MCLS::SolverManager<Vector,Matrix> > solver_manager =
	factory.create( solver_type, mcls_list );

    // Solve the problem.
    solver_manager->setProblem( linear_problem );
    solver_manager->solve();

    // Output final timing.
    Teuchos::TableFormat &format = Teuchos::TimeMonitor::format();
    format.setPrecision(5);
    Teuchos::TimeMonitor::summarize();

    return 0;
}

//---------------------------------------------------------------------------//
// end neutron_diffusion.cpp
//---------------------------------------------------------------------------//

