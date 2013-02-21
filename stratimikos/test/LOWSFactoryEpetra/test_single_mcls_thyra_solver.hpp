
#ifndef TEST_SINGLE_MCLS_THYRA_SOLVER_HPP
#define TEST_SINGLE_MCLS_THYRA_SOLVER_HPP

#include "Teuchos_FancyOStream.hpp"

namespace Teuchos { class ParameterList; }

namespace Thyra {

/** \brief Testing function for a single mcls solver with a single matrix.
 *
 */
bool test_single_mcls_thyra_solver(
  const std::string                       matrixFile
  ,const bool                             useJacobi
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
  ,Teuchos::FancyOStream                  *out
  );

} // namespace Thyra

#endif // TEST_SINGLE_MCLS_THYRA_SOLVER_HPP
