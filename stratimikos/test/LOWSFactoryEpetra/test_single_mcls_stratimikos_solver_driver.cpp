
#include "test_single_mcls_stratimikos_solver.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_StandardCatchMacros.hpp"


int main(int argc, char* argv[])
{
  
  using Teuchos::CommandLineProcessor;

  bool success = true;
  bool verbose = true;

  Teuchos::GlobalMPISession mpiSession(&argc, &argv);

  Teuchos::RCP<Teuchos::FancyOStream>
    out = Teuchos::VerboseObjectBase::getDefaultOStream();

  try {

    //
    // Read options from command-line
    //
    
    std::string     matrixFile             = "";
    bool            useScaling              = false;
    bool            testTranspose          = false;
    bool            usePreconditioner      = false;
    int             numRhs                 = 1;
    int             numRandomVectors       = 1;
    double          maxFwdError            = 1e-14;
    int             maxIterations          = 400;
    int             outputFrequency        = 10;
    bool            outputMaxResOnly       = true;
    double          maxResid               = 1e-6;
    double          maxSolutionError       = 1e-6;
    bool            showAllTests           = false;
    bool            dumpAll                = false;
    double          weightCutoff           = 1e-4;
    int             mcCheckFrequency       = 1024;
    int             mcBufferSize           = 1024;
    bool            reproducibleMC         = true;
    int             overlapSize            = 0;
    int             numSets                = 1;
    int             numHistories           = 1000;
    std::string     mcType                 = "Adjoint";
    int             blockSize              = 1;
    std::string     solverType             = "MCSA";
    std::string     precType               = "Point Jacobi";

    CommandLineProcessor  clp;
    clp.throwExceptions(false);
    clp.addOutputSetupOptions(true);
    clp.setOption( "matrix-file", &matrixFile, "Matrix input file [Required]." );
    clp.setOption( "use-scaling", "no-use-scaling", &useScaling, "Use Scaling preconditioning or not" );
    clp.setOption( "test-transpose", "no-test-transpose", &testTranspose, "Test the transpose solve or not." );
    clp.setOption( "use-preconditioner", "no-use-preconditioner", &usePreconditioner, "Use the preconditioner or not." );
    clp.setOption( "num-rhs", &numRhs, "Number of RHS in linear solve." );
    clp.setOption( "num-random-vectors", &numRandomVectors, "Number of times a test is performed with different random vectors." );
    clp.setOption( "max-fwd-error", &maxFwdError, "The maximum relative error in the forward operator." );
    clp.setOption( "max-iters", &maxIterations, "The maximum number of linear solver iterations to take." );
    clp.setOption( "output-frequency", &outputFrequency, "Number of linear solver iterations between output" );
    clp.setOption( "output-max-res-only", "output-all-res", &outputMaxResOnly, 
		   "Determines if only the max residual is printed or if all residuals are printed per iteration." );
    clp.setOption( "max-resid", &maxResid, "The maximum relative error in the residual." );
    clp.setOption( "max-solution-error", &maxSolutionError, "The maximum relative error in the solution of the linear system." );
    clp.setOption( "verbose", "quiet", &verbose, "Set if output is printed or not." );
    clp.setOption( "show-all-tests", "no-show-all-tests", &showAllTests, "Set if all the tests are shown or not." );
    clp.setOption( "dump-all", "no-dump-all", &dumpAll, "Determines if vectors are printed or not." );
    clp.setOption( "mc-type", &mcType, "Determines underlying MC type in solver." );
    clp.setOption( "mc-cutoff", &weightCutoff, "Determines underlying MC history weight cutoff." );
    clp.setOption( "mc-frequency", &mcCheckFrequency, "Determines the frequency of MC communications." );
    clp.setOption( "mc-buffer-size", &mcBufferSize, "Determines the size of MC history buffers." );
    clp.setOption( "mc-reproduce", "reproducible", &reproducibleMC, 
		   "Determines whether or not to communicate RNG with MC histories." );
    clp.setOption( "mc-overlap", &overlapSize, "Determines the size of overlap in MC problem." );
    clp.setOption( "mc-sets", &numSets, "Determines the number of sets in the MC problem." );
    clp.setOption( "mc-histories", &numHistories, "Determines the number of histories in the MC problem." );
    clp.setOption( "block-size", &blockSize, "Block Jacobi preconditioning block size." );
    clp.setOption( "solver-type", &solverType, "Determines MCLS solver." );
    clp.setOption( "prec-type", &precType, "Determines MCLS preconditioner." );
    CommandLineProcessor::EParseCommandLineReturn parse_return = clp.parse(argc,argv);
    if( parse_return != CommandLineProcessor::PARSE_SUCCESSFUL ) return parse_return;

    TEUCHOS_TEST_FOR_EXCEPT( matrixFile == "" );

    Teuchos::ParameterList mclsLOWSFPL;
    mclsLOWSFPL.set("Solver Type",solverType);

    Teuchos::ParameterList& mclsLOWSFPL_solver =
      mclsLOWSFPL.sublist("Solver Types");

    Teuchos::ParameterList& mclsLOWSFPL_mcsa =
      mclsLOWSFPL_solver.sublist("MCSA");
    mclsLOWSFPL_mcsa.set("Maximum Iterations",int(maxIterations));
    mclsLOWSFPL_mcsa.set("Convergence Tolerance",double(maxResid));
    mclsLOWSFPL_mcsa.set("MC Type",std::string(mcType));
    mclsLOWSFPL_mcsa.set("Iteration Print Frequency",int(outputFrequency));
    mclsLOWSFPL_mcsa.set("Weight Cutoff",double(weightCutoff));
    mclsLOWSFPL_mcsa.set("MC Check Frequency",int(mcCheckFrequency));
    mclsLOWSFPL_mcsa.set("MC Buffer Size",int(mcBufferSize));
    mclsLOWSFPL_mcsa.set("Reproducible MC Mode",bool(reproducibleMC));
    mclsLOWSFPL_mcsa.set("Overlap Size",int(overlapSize));
    mclsLOWSFPL_mcsa.set("Number of Sets",int(numSets));
    mclsLOWSFPL_mcsa.set("Set Number of Histories",int(numHistories));

    Teuchos::ParameterList& mclsLOWSFPL_seqmc =
      mclsLOWSFPL_solver.sublist("Sequential MC");
    mclsLOWSFPL_seqmc.set("Maximum Iterations",int(maxIterations));
    mclsLOWSFPL_seqmc.set("Convergence Tolerance",double(maxResid));
    mclsLOWSFPL_seqmc.set("MC Type",std::string(mcType));
    mclsLOWSFPL_seqmc.set("Iteration Print Frequency",int(outputFrequency));
    mclsLOWSFPL_seqmc.set("Weight Cutoff",double(weightCutoff));
    mclsLOWSFPL_seqmc.set("MC Check Frequency",int(mcCheckFrequency));
    mclsLOWSFPL_seqmc.set("MC Buffer Size",int(mcBufferSize));
    mclsLOWSFPL_seqmc.set("Reproducible MC Mode",bool(reproducibleMC));
    mclsLOWSFPL_seqmc.set("Overlap Size",int(overlapSize));
    mclsLOWSFPL_seqmc.set("Number of Sets",int(numSets));
    mclsLOWSFPL_seqmc.set("Set Number of Histories",int(numHistories));

    Teuchos::ParameterList& mclsLOWSFPL_adjmc =
	mclsLOWSFPL_solver.sublist("Adjoint MC");
    mclsLOWSFPL_adjmc.set("Convergence Tolerance",double(maxResid));
    mclsLOWSFPL_adjmc.set("Weight Cutoff",double(weightCutoff));
    mclsLOWSFPL_adjmc.set("MC Check Frequency",int(mcCheckFrequency));
    mclsLOWSFPL_adjmc.set("MC Buffer Size",int(mcBufferSize));
    mclsLOWSFPL_adjmc.set("Reproducible MC Mode",bool(reproducibleMC));
    mclsLOWSFPL_adjmc.set("Overlap Size",int(overlapSize));
    mclsLOWSFPL_adjmc.set("Number of Sets",int(numSets));
    mclsLOWSFPL_adjmc.set("Set Number of Histories",int(numHistories));

    if(usePreconditioner) 
    {
	Teuchos::ParameterList& precPL = mclsLOWSFPL.sublist("Preconditioner Type");
	precPL.set("Preconditioner Type",precType);

	Teuchos::ParameterList& precPL_prec = precPL.sublist("Preconditioner Types");

	Teuchos::ParameterList& precPL_bj = precPL_prec.sublist("Block Jacobi");
	precPL_bj.set("Jacobi Block Size", blockSize);
    }

    success
      = Thyra::test_single_mcls_stratimikos_solver(
	  matrixFile,useScaling,testTranspose,usePreconditioner,numRhs,numRandomVectors
        ,maxFwdError,maxResid,maxSolutionError,showAllTests,dumpAll
        ,&mclsLOWSFPL,&mclsLOWSFPL
        ,verbose?&*out:0
        );

  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success)
  
  if (verbose) {
    if(success)  *out << "\nCongratulations! All of the tests checked out!\n";
    else         *out << "\nOh no! At least one of the tests failed!\n";
  }

  return ( success ? 0 : 1 );
}
