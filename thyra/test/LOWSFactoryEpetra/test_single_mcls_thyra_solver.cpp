#include "test_single_mcls_thyra_solver.hpp"

#include "Thyra_MCLSLinearOpWithSolveFactory.hpp"
#include "Thyra_MCLSPreconditionerFactory.hpp"
#include "Thyra_LinearOpWithSolveFactoryHelpers.hpp"
#include "Thyra_EpetraLinearOp.hpp"
#include "Thyra_LinearOpTester.hpp"
#include "Thyra_LinearOpWithSolveBase.hpp"
#include "Thyra_LinearOpWithSolveTester.hpp"
#include "Thyra_MultiVectorStdOps.hpp"
#include "Thyra_VectorStdOps.hpp"
#include "EpetraExt_readEpetraLinearSystem.h"
#include "Epetra_Comm.h"
#include "Epetra_SerialComm.h"
#include "Epetra_MpiComm.h"
#include "Epetra_Vector.h"
#include "Epetra_CrsMatrix.h"
#include "Teuchos_ParameterList.hpp"

bool Thyra::test_single_mcls_thyra_solver(
  const std::string                       matrixFile
  ,const bool                             testTranspose
  ,const bool                             usePreconditioner
  ,const int                              numRhs
  ,const int                              numRandomVectors
  ,const double                           maxFwdError
  ,const double                           maxResid
  ,const double                           maxSolutionError
  ,const bool                             showAllTests
  ,const bool                             dumpAll
  ,Teuchos::ParameterList                 *mclsLOWSFPL
  ,Teuchos::ParameterList                 *precPL
  ,Teuchos::FancyOStream                  *out_arg
  )
{
  using Teuchos::rcp;
  using Teuchos::OSTab;
  bool result, success = true;

  Teuchos::RCP<Teuchos::FancyOStream> out = Teuchos::rcp(out_arg,false);

  try {

    if(out.get()) {
      *out << "\n***"
           << "\n*** Testing Thyra::MCLSLinearOpWithSolveFactory (and Thyra::MCLSLinearOpWithSolve)"
           << "\n***\n"
           << "\nEchoing input options:"
           << "\n  matrixFile             = " << matrixFile
           << "\n  testTranspose          = " << testTranspose
           << "\n  usePreconditioner      = " << usePreconditioner
           << "\n  numRhs                 = " << numRhs
           << "\n  numRandomVectors       = " << numRandomVectors
           << "\n  maxFwdError            = " << maxFwdError
           << "\n  maxResid               = " << maxResid
           << "\n  showAllTests           = " << showAllTests
           << "\n  dumpAll                = " << dumpAll
           << std::endl;
    }

    if(out.get()) *out << "\nA) Reading in an epetra matrix A from the file \'"<<matrixFile<<"\' ...\n";
  
    Teuchos::RCP<Epetra_Comm> comm;
#ifdef HAVE_MPI
    comm = Teuchos::rcp(new Epetra_MpiComm(MPI_COMM_WORLD));
#else
    comm = Teuchos::rcp(new Epetra_SerialComm());
#endif

    Teuchos::RCP<Epetra_CrsMatrix> epetra_A;
    EpetraExt::readEpetraLinearSystem( matrixFile, *comm, &epetra_A );

    Teuchos::RCP<const LinearOpBase<double> > A = epetraLinearOp(epetra_A);

    if(out.get() && dumpAll) *out << "\ndescribe(A) =\n" << describe(*A,Teuchos::VERB_EXTREME);

    if(out.get()) *out << "\nB) Creating a MCLSLinearOpWithSolveFactory object opFactory ...\n";

    Teuchos::RCP<LinearOpWithSolveFactoryBase<double> > lowsFactory;
    {
      Teuchos::RCP<MCLSLinearOpWithSolveFactory<double> >
        mclsLowsFactory = Teuchos::rcp(new MCLSLinearOpWithSolveFactory<double>());
      lowsFactory = mclsLowsFactory;
    }

    if(out.get()) {
      *out << "\nlowsFactory.getValidParameters() before setting preconditioner factory:\n";
      lowsFactory->getValidParameters()->print(OSTab(out).o(),0,true,false);
    }

    if(usePreconditioner) 
    {
      if(out.get()) {
        *out << "\nSetting an MCLS preconditioner factory ...\n";
      }
      RCP<PreconditionerFactoryBase<double> >
      precFactory = Teuchos::rcp(new MCLSPreconditionerFactory<double>());
      if (precPL)
        precFactory->setParameterList(rcp(precPL,false));
      lowsFactory->setPreconditionerFactory(precFactory,"MCLS");
    }

    if(out.get()) {
      *out << "\nlowsFactory.getValidParameters() after setting preconditioner factory:\n";
      lowsFactory->getValidParameters()->print(OSTab(out).o(),0,true,false);
      *out << "\nmclsLOWSFPL before setting parameters:\n";
      mclsLOWSFPL->print(OSTab(out).o(),0,true);
    }

    lowsFactory->setParameterList(Teuchos::rcp(mclsLOWSFPL,false));

    if(out.get()) {
      *out << "\nmclsLOWSFPL after setting parameters:\n";
      mclsLOWSFPL->print(OSTab(out).o(),0,true);
    }

    if(out.get()) *out << "\nC) Creating a MCLSLinearOpWithSolve object nsA from A ...\n";

    Teuchos::RCP<LinearOpWithSolveBase<double> > nsA = lowsFactory->createOp();
    Thyra::initializeOp<double>(*lowsFactory,  A, nsA.ptr());

    if(out.get()) *out << "\nD) Testing the LinearOpBase interface of nsA ...\n";

    LinearOpTester<double> linearOpTester;
    linearOpTester.check_adjoint(testTranspose);
    linearOpTester.num_rhs(numRhs);
    linearOpTester.num_random_vectors(numRandomVectors);
    linearOpTester.set_all_error_tol(maxFwdError);
    linearOpTester.set_all_warning_tol(1e-2*maxFwdError);
    linearOpTester.show_all_tests(showAllTests);
    linearOpTester.dump_all(dumpAll);
    Thyra::seed_randomize<double>(0);
    result = linearOpTester.check(*nsA,Teuchos::Ptr<Teuchos::FancyOStream>(out.get()));
    if(!result) success = false;

    if(out.get()) *out << "\nE) Testing the LinearOpWithSolveBase interface of nsA ...\n";
    
    LinearOpWithSolveTester<double> linearOpWithSolveTester;
    linearOpWithSolveTester.num_rhs(numRhs);
    linearOpWithSolveTester.turn_off_all_tests();
    linearOpWithSolveTester.check_forward_default(true);
    linearOpWithSolveTester.check_forward_residual(true);
    if(testTranspose) {
      linearOpWithSolveTester.check_adjoint_default(true);
      linearOpWithSolveTester.check_adjoint_residual(true);
    }
    else {
      linearOpWithSolveTester.check_adjoint_default(false);
      linearOpWithSolveTester.check_adjoint_residual(false);
    }
    linearOpWithSolveTester.set_all_solve_tol(maxResid);
    linearOpWithSolveTester.set_all_slack_error_tol(maxResid);
    linearOpWithSolveTester.set_all_slack_warning_tol(1e+1*maxResid);
    linearOpWithSolveTester.forward_default_residual_error_tol(2*maxResid);
    linearOpWithSolveTester.forward_default_solution_error_error_tol(maxSolutionError);
    linearOpWithSolveTester.adjoint_default_residual_error_tol(2*maxResid);
    linearOpWithSolveTester.adjoint_default_solution_error_error_tol(maxSolutionError);
    linearOpWithSolveTester.show_all_tests(showAllTests);
    linearOpWithSolveTester.dump_all(dumpAll);
    Thyra::seed_randomize<double>(0);
    result = linearOpWithSolveTester.check(*nsA,out.get());
    if(!result) success = false;

    if(out.get()) {
      *out << "\nmclsLOWSFPL after solving:\n";
      mclsLOWSFPL->print(OSTab(out).o(),0,true);
    }
    
  }
  catch( const std::exception &excpt ) {
    if(out.get()) *out << std::flush;
    std::cerr << "*** Caught standard exception : " << excpt.what() << std::endl;
    success = false;
  }
   
  return success;
    
}
