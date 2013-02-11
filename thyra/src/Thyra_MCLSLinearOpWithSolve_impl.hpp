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
 * \file Thyra_MCLSLinearOpWithSolve_impl.hpp
 * \author Stuart R. Slattery
 * \brief Thyra LinearOpWithSolve implementation for MCLS.
 */
//---------------------------------------------------------------------------//

#ifndef THYRA_MCLS_LINEAR_OP_WITH_SOLVE_IMPL_HPP
#define THYRA_MCLS_LINEAR_OP_WITH_SOLVE_IMPL_HPP

#include <MCLS_DBC.hpp>

#include <Teuchos_TimeMonitor.hpp>

#include <Thyra_LinearOpWithSolveHelpers.hpp>

namespace Thyra
{

//---------------------------------------------------------------------------//
/*!
 * \brief Uninitialized constructor.
 */
template<class Scalar>
MCLSLinearOpWithSolve<Scalar>::MCLSLinearOpWithSolve()
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Initializes given precreated solver objects. 
 */
template<class Vector, class MultiVector, class Matrix>
void MCLSLinearOpWithSolve<Scalar>::initialize(
    const RCP<MCLS::LinearProblemAdapter<Vector,MultiVector,Matrix> >& linear_problem,
    const RCP<Teuchos::ParameterList>& plist,
    const RCP<MCLS::SolverManagerAdapter<Vector,MultiVector,Matrix> >& solver,
    const RCP<const LinearOpSourceBase<Scalar> >& fwd_op_src,
    const RCP<const PreconditionerBase<Scalar> >& prec,
    const bool is_external_prec,
    const RCP<const LinearOpSourceBase<Scalar> >& approx_fwd_op_src,
    const ESupportSolveUse& support_solve_use )
    : d_linear_problem( linear_problem )
    , d_plist( plist )
    , d_solver( solver )
    , d_fwd_op_src( fwd_op_src )
    , d_prec( prec )
    , d_is_external_prec( is_external_prec )
    , d_support_solve_use( support_solve_use )
{
    Require( nonnull(d_solver) );

    if ( nonnull(d_plist) )
    {
	if ( d_plist->isParameter("Convergence Tolerance") )
	{
	    d_default_tol = d_plist->get<double>("Convergence Tolerance");
	}
	else
	{
	    d_default_tol = 
		d_solver->getValidParameters()->d_plist->get<double>(
		    "Convergence Tolerance");
	}
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Extract the forward <tt>LinearOpBase<Scalar></tt> object so that it
 * can be modified. 
 */
template<class Scalar>
RCP<const LinearOpSourceBase<Scalar> > MCLSLinearOpWithSolve<Scalar>::extract_fwdOpSrc()
{
    RCP<const LinearOpSourceBaseScalar> fwd_op_src = d_fwd_op_src;
    d_fwd_op_src = Teuchos::null;
    return fwd_op_src;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Extract the forward preconditioner object so that it can be
 * modified.
 */
template<class Scalar>
RCP<const LinearOpSourceBase<Scalar> > MCLSLinearOpWithSolve<Scalar>::extract_prec()
{
    RCP<const LinearOpSourceBaseScalar> prec = d_prec;
    d_prec = Teuchos::null;
    return prec;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if the preconditioner is external or not.
 */
template<class Scalar>
boll MCLSLinearOpWithSolve<Scalar>::isExternalPrec() const
{
    return d_is_external_prec;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Extract the approximate forward <tt>LinearOpBase<Scalar></tt> object
 * so that it can be modified.
 */
template<class Scalar>
RCP<const LinearOpSourceBase<Scalar> > 
MCLSLinearOpWithSolve<Scalar>::extract_approxFwdOpSrc()
{
    RCP<const LinearOpSourceBaseScalar> approx_fwd_op_src = 
	d_approx_fwd_op_src;
    d_approx_fwd_op_src = Teuchos::null;
    return approx_fwd_op_src;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Check for support.
 */
template<class Scalar>
ESupportSolveUse MCLSLinearOpWithSolve<Scalar>::supportSolveUse()
{
    return d_support_solve_use;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Uninitializes and returns stored quantities.
 */
template<class Scalar>
void MCLSLinearOpWithSolve<Scalar>::uninitialize(
    RCP<MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> > *lp,
    RCP<Teuchos::ParameterList> *solverPL,
    RCP<MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> > *iterativeSolver,
    RCP<const LinearOpSourceBase<Scalar> > *fwdOpSrc,
    RCP<const PreconditionerBase<Scalar> > *prec,
    bool *isExternalPrec,
    RCP<const LinearOpSourceBase<Scalar> > *approxFwdOpSrc,
    ESupportSolveUse *supportSolveUse
    )
{
    if (lp) *lp = lp_;
    if (solverPL) *solverPL = solverPL_;
    if (iterativeSolver) *iterativeSolver = iterativeSolver_;
    if (fwdOpSrc) *fwdOpSrc = fwdOpSrc_;
    if (prec) *prec = prec_;
    if (isExternalPrec) *isExternalPrec = isExternalPrec_;
    if (approxFwdOpSrc) *approxFwdOpSrc = approxFwdOpSrc_;
    if (supportSolveUse) *supportSolveUse = supportSolveUse_;

    lp_ = Teuchos::null;
    solverPL_ = Teuchos::null;
    iterativeSolver_ = Teuchos::null;
    fwdOpSrc_ = Teuchos::null;
    prec_ = Teuchos::null;
    isExternalPrec_ = false;
    approxFwdOpSrc_ = Teuchos::null;
    supportSolveUse_ = SUPPORT_SOLVE_UNSPECIFIED;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the range of the operator for the linear problem being solved by
 * this solver.
 */
template<class Scalar>
RCP<const VectorSpaceBase<Scalar> > MCLSLinearOpWithSolve<Scalar>::range() const
{
    if ( nonnull(d_linear_problem) )
    {
	return d_linear_problem->getOperator()->range();
    }

    return Teuchos::null;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the domain of the operator for the linear problem being solved by
 * this solver.
 */
template<class Scalar>
RCP<const VectorSpaceBase<Scalar> > MCLSLinearOpWithSolve<Scalar>::domain() const
{
    if ( nonnull(d_linear_problem) ) 
    {
	return d_linear_problem->getOperator()->domain();
    }

    return Teuchos::null;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Clone the operator for the linear problem being solved by this
 * solver.
 */
template<class Scalar>
RCP<const VectorSpaceBase<Scalar> > MCLSLinearOpWithSolve<Scalar>::clone() const
{
    // Not supported.
    return Teuchos::null;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a description of this object.
 */
template<class Scalar>
std::string MCLSLinearOpWithSolve<Scalar>::description() const
{
    std::ostringstream oss;
    oss << Teuchos::Describable::description();

    if ( nonnull(d_linear_problem) && 
	 nonnull(d_linear_problem->getOperator()) ) 
    {
	oss << "{";
	oss << "iterativeSolver=\'" << d_solver->description()<<"\'";
	oss << ",fwdOp=\'" << d_linear_problem->getOperator()->description()<<"\'";
	oss << "}";
    }

    return oss.str();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Describe this object.
 */
template<class Scalar>
void MCLSLinearOpWithSolve<Scalar>::describe( 
    Teuchos::FancyOStream &out_arg,
    const Teuchos::EVerbosityLevel verbLevel ) const
{
    typedef Teuchos::ScalarTraits<Scalar> ST;
    using Teuchos::FancyOStream;
    using Teuchos::OSTab;
    using Teuchos::describe;
    RCP<FancyOStream> out = rcp(&out_arg,false);
    OSTab tab(out);
    switch (verbLevel) {
	case Teuchos::VERB_DEFAULT:
	case Teuchos::VERB_LOW:
	    *out << this->description() << std::endl;
	    break;
	case Teuchos::VERB_MEDIUM:
	case Teuchos::VERB_HIGH:
	case Teuchos::VERB_EXTREME:
	{
	    *out << Teuchos::Describable::description() << "{"
		 << "rangeDim=" << this->range()->dim()
		 << ",domainDim=" << this->domain()->dim() << "}\n";

	    if ( nonnull(d_linear_problem->getOperator()) ) 
	    {
		OSTab tab( out );
		*out << "iterativeSolver = "<< describe(*d_solver,verbLevel)
		     << "fwdOp = " 
		     << describe(*d_linear_problem->getOperator(),verbLevel);
	    }
	    break;
	}
	default:
	    Insist( false, "Incorrect verbosity level provided" );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether a given operator transpose type is supported.
 */
template<class Scalar>
bool MCLSLinearOpWithSolve<Scalar>::opSupportedImpl( const EOpTransp M_trans )
{
    return ::Thyra::opSupported( d_linear_problem->getOperator(), M_trans );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the linear operator from the linear problem being solved by
 * this solver to a multivector.
 */
template<class Scalar>
void MCLSLinearOpWithSolve<Scalar>::applyImpl( const EOpTransp M_trans,
				       const MultiVectorBase<Scalar> &X,
				       const Ptr<MultiVectorBase<Scalar> > &Y,
				       const Scalar alpha,
				       const Scalar beta ) const
{
    ::Thyra::apply<Scalar>(
	d_linear_problem->getOperator(), M_trans, X, Y, alpha, beta );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether this solver supports the given transpose type.
 */
template<class Scalar>
bool MCLSLinearOpWithSolve<Scalar>::solveSupportsImpl( const EOpTransp M_trans )
{
    return solveSupportsNewImpl( M_trans, Teuchos::null );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether this solver supports the given transpose type and
 * solve criteria.
 */
template<class Scalar>
bool MCLSLinearOpWithSolve<Scalar>::solveSupportsNewImpl( 
    const EOpTransp M_trans,
    const Ptr<const SolveCriteria<Scalar> > solveCriteria ) const
{
    // Only forward solves are currently supported.
    if ( transp == NOTRANS )
    {
	// Only residual scaled by rhs supported.
	return ( solveCriteria->solveMeasureType.useDefault() ||
		 solveCriteria->solveMeasureType(SOLVE_MEASURE_NORM_RESIDUAL,
						 SOLVE_MEASURE_NORM_RHS) );
    }

    return false;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether this solver supports the given transpose type and
    solve measure type. 
*/
template<class Scalar>
bool MCLSLinearOpWithSolve<Scalar>::solveSupportsSolveMeasureTypeImpl(
    EOpTransp M_trans, const SolveMeasureType& solveMeasureType ) const
{
  SolveCriteria<Scalar> solveCriteria(
      solveMeasureType, SolveCriteria<Scalar>::unspecifiedTolerance() );
  return solveSupportsNewImpl( M_trans, Teuchos::constOptInArg(solveCriteria) );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Solve the linear problem.
*/
template<class Scalar>
SolveStatus<Scalar> MCLSLinearOpWithSolve<Scalar>::solveImpl(
  const EOpTransp M_trans,
  const MultiVectorBase<Scalar> &B,
  const Ptr<MultiVectorBase<Scalar> > &X,
  const Ptr<const SolveCriteria<Scalar> > solveCriteria ) const
{
    // Setup timing.
    THYRA_FUNC_TIME_MONITOR("Stratimikos: MCLSLOWS");
    Teuchos::Time total_timer("");
    total_timer.start(true);

    // Validate input.
    Insist( this->solveSupportsImpl(M_trans), 
	    "Solve does not support transpose type." );

    // Output before solve.
    RCP<Teuchos::FancyOStream> out = this->getOStream();
    Teuchos::EVerbosityLevel verbLevel = this->getVerbLevel();
    Teuchos::OSTab tab = this->getOSTab();
    if ( out.get() && 
	 static_cast<int>(verbLevel) > static_cast<int>(Teuchos::VERB_NONE) )
    {
	*out << "\nSolving block system using MCLS ...\n\n";
    }

    // Parameter list for the current solve.
    const RCP<ParameterList> tmp_pl = Teuchos::parameterList();

    // The solver's valid parameter list.
    RCP<const ParameterList> valid_pl = d_solver->getValidParameters();

    // Set solve criteria.
    SolveMeasureType solve_measure;
    if ( nonnull(solveCriteria) )
    {
	// Get the solve measure.
	solve_measure = solveCriteria->solveMeasureType;

	// Set convergence tolerance.
	typename Teuchos::ScalarTraits<Scalar>::magnitudeType requested_tol =
	    solveCriteria->requestedTol;
	if ( solve_measure.useDefault() )
	{
	    tmp_pl->set("Convergence Tolerance", d_default_tol);
	}
	else if (solve_measure(SOLVE_MEASURE_NORM_RESIDUAL, SOLVE_MEASURE_NORM_RHS)) 
	{
	    if (requested_tol != SolveCriteria<Scalar>::unspecifiedTolerance()) 
	    {
		tmp_pl->set("Convergence Tolerance", requested_tol);
	    }
	    else {
		tmp_pl->set("Convergence Tolerance", default_tol);
	    }
	    setResidualScalingType (tmpPL, validPL, "Norm of RHS");
	}
	else
	{
	    temp_pl->set("Convergence Tolerance", 1.0);
	}

	// Set the maximum number of iterations.
	if ( nonnull(solveCriteria->extraParameters) ) 
	{
	    if (Teuchos::isParameterType<int>(*solveCriteria->extraParameters,
					      "Maximum Iterations") ) 
	    {
		tmp_pl->set("Max Number of Iterations", 
			    Teuchos::get<int>(*solveCriteria->extraParameters,
					      "Maximum Iterations") );
	    }
	}
    }
    else
    {
	tmp_pl->set("Convergence Tolerance", d_default_tol);
    }

    // Set the parameters with the solver.
    d_solver->setParameters( tmp_pl );

    // Set the problem.
    d_linear_problem->setLHS( Teuchos::rcpFromPtr(X) );
    d_linear_problem->setRHS( Teuchos::rcpFromRef(B) );

    // Solve the linear system.
    SolveStatus<Scalar> status = d_solver->solve();
    total_timer.stop();

    // Report the overall timing.
    if ( out.get() && 
	 static_cast<int>(verbLevel) >= static_cast<int>(Teuchos::VERB_LOW) )
    {
	*out << "\nTotal solve time = "
	     << total_timer.totalElapsedTime() <<" sec\n";
    }

    // Return the solve status.
    return status;
}

//---------------------------------------------------------------------------//

} // end namespace Thyra

//---------------------------------------------------------------------------//

#endif // end THYRA_MCLS_LINEAR_OP_WITH_SOLVE_IMPL_HPP

//---------------------------------------------------------------------------//
// end Thyra_MCLSLinearOpWithSolve.hpp
//---------------------------------------------------------------------------//

