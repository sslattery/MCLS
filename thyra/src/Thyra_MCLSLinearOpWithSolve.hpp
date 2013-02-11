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
 * \file Thyra_MCLSLinearOpWithSolve.hpp
 * \author Stuart R. Slattery
 * \brief Thyra LinearOpWithSolve implementation for MCLS.
 */
//---------------------------------------------------------------------------//

#ifndef THYRA_MCLS_LINEAR_OP_WITH_SOLVE_HPP
#define THYRA_MCLS_LINEAR_OP_WITH_SOLVE_HPP

#include "MCLS_LinearProblemAdapter.hpp"
#include "MCLS_SolverManagerAdapter.hpp"

#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpSourceBase.hpp>

namespace Thyra
{

//---------------------------------------------------------------------------//
/*!
 * \class MCLSLinearOpWithSolve
 * \brief Thyra::LinearOpWithSolve implementation for MCLS solvers.
 */
template<class Scalar>
class MCLSLinearOpWithSolve : virtual public LinearOpWithSolveBase<Scalar>
{
  public:

    //! Typedefs.
    typedef MultiVectorBase<Scalar>                  MV_t;
    typedef LinearOpBase<Scalar>                     LO_t;
    //@}

    // Uninitialized constructor.
    MCLSLinearOpWithSolve();

    // Initializes given precreated solver objects.
    void initialize(
	const RCP<MCLS::LinearProblemBase<Scalar> >& linear_problem,
	const RCP<Teuchos::ParameterList>& plist,
	const RCP<MCLS::SolverManagerBase<Scalar> >&solver,
	const RCP<const LinearOpSourceBase<Scalar> >& fwd_op_src,
	const RCP<const PreconditionerBase<Scalar> >& prec,
	const bool is_external_prec,
	const RCP<const LinearOpSourceBase<Scalar> >& approx_fwd_op_src,
	const ESupportSolveUse& support_solve_use );

    /** @name Extraction methods */
    //@{ 
    // Extract the forward <tt>LinearOpSourceBase<Scalar></tt> object so that
    // it can be modified. 
    RCP<const LinearOpSourceBase<Scalar> > extract_fwdOpSrc();

    // Extract the preconditioner.
    RCP<const PreconditionerBase<Scalar> > extract_prec();

    // Determine if the preconditioner was external or not.
    bool isExternalPrec() const;

    // Extract the approximate forward <tt>LinearOpSourceBase<Scalar></tt>
    // object so that it can be modified.
    RCP<const LinearOpSourceBase<Scalar> > extract_approxFwdOpSrc();

    // Check for support.
    ESupportSolveUse supportSolveUse() const;

    // Uninitializes and returns stored quantities.
    void uninitialize(
	RCP<MCLS::LinearProblemBase<Scalar> > *lp = NULL,
	RCP<Teuchos::ParameterList> *solverPL = NULL,
	RCP<MCLS::SolverManagerBase<Scalar> > *iterativeSolver = NULL,
	RCP<const LinearOpSourceBase<Scalar> > *fwdOpSrc = NULL,
	RCP<const PreconditionerBase<Scalar> > *prec = NULL,
	bool *isExternalPrec = NULL,
	RCP<const LinearOpSourceBase<Scalar> > *approxFwdOpSrc = NULL,
	ESupportSolveUse *supportSolveUse = NULL
	);
    //@}


    /** @name Overridden from LinearOpBase */
    //@{
    // Get the range of the operator for the linear problem being solved by
    // this solver. 
    RCP<const VectorSpaceBase<Scalar> > range() const;

    // Get the domain of the operator for the linear problem being solved by
    // this solver. 
    RCP<const VectorSpaceBase<Scalar> > domain() const;

    // Clone the operator for the linear problem being solved by this solver. 
    RCP<const LinearOpBase<Scalar> > clone() const;
    //@}

    /** @name Overridden from Teuchos::Describable */
    //@{
    // Get a description of this object.
    std::string description() const;

    // Describe this object.
    void describe(
	Teuchos::FancyOStream &out,
	const Teuchos::EVerbosityLevel verbLevel
	) const;
    //@}

  protected: 

    /** @name Overridden from LinearOpBase  */
    //@{
    // Return whether a given operator transpose type is supported.
    virtual bool opSupportedImpl( EOpTransp M_trans ) const;

    // Apply the linear operator from the linear problem being solved by this
    // solver to a multivector.
    virtual void applyImpl( const EOpTransp M_trans,
			    const MultiVectorBase<Scalar> &X,
			    const Ptr<MultiVectorBase<Scalar> > &Y,
			    const Scalar alpha,
			    const Scalar beta ) const;
    //@}

    /** @name Overridden from LinearOpWithSolveBase. */
    //@{
    // Return whether this solver supports the given transpose type.
    virtual bool solveSupportsImpl( EOpTransp M_trans ) const;

    // Return whether this solver supports the given transpose type and solve
    // criteria. 
    virtual bool solveSupportsNewImpl(
	EOpTransp transp,
	const Ptr<const SolveCriteria<Scalar> > solveCriteria ) const;

    // Return whether this solver supports the given transpose type and solve
    // measure type.
    virtual bool solveSupportsSolveMeasureTypeImpl(
	EOpTransp M_trans, const SolveMeasureType& solveMeasureType ) const;

    // Solve the linear problem.
    virtual SolveStatus<Scalar> solveImpl(
	const EOpTransp M_trans,
	const MultiVectorBase<Scalar> &B,
	const Ptr<MultiVectorBase<Scalar> > &X,
	const Ptr<const SolveCriteria<Scalar> > solveCriteria ) const;
    //@}

  private:

    // Blocked linear problem.
    RCP<MCLS::LinearProblemBase<Scalar> > d_linear_problem;

    // Solver parameters.
    RCP<Teuchos::ParameterList> d_plist;

    // Blocked solver manager.
    RCP<MCLS::SolverManagerBase<Scalar> > d_solver;

    // Linear operator source.
    RCP<const LinearOpSourceBase<Scalar> > d_fwd_op_src;

    // Preconditioner.
    RCP<const PreconditionerBase<Scalar> > d_prec;

    // Approximate linear operator source.
    RCP<const LinearOpSourceBase<Scalar> > d_approx_fwd_op_src;

    // External preconditioner status.
    bool d_is_external_prec;

    // Support status.
    ESupportSolveUse d_support_solve_use;

    // Default convergence tolerance.
    typename Teuchos::ScalarTraits<Scalar>::magnitudeType d_default_tol;
};

//---------------------------------------------------------------------------//

} // end namespace Thyra

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "Thyra_MCLSLinearOpWithSolve_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end THYRA_MCLS_LINEAR_OP_WITH_SOLVE_HPP

//---------------------------------------------------------------------------//
// end Thyra_MCLSLinearOpWithSolve.hpp
//---------------------------------------------------------------------------//

