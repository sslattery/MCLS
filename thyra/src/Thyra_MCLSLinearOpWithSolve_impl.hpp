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

#include <Thyra_LinearOpWithSolveHelpers.hpp>

namespace Thyra
{

//---------------------------------------------------------------------------//
/*!
 * \brief Uninitialized constructor.
 */
template<class Scalar>
MCLSLinearOpWithSolve::MCLSLinearOpWithSolve()
{ /* ... */ }

//---------------------------------------------------------------------------//
/*!
 * \brief Initializes given precreated solver objects. 
 */
template<class Scalar>
void MCLSLinearOpWithSolve::initialize(
    const RCP<MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> >& linear_problem,
    const RCP<Teuchos::ParameterList>& plist,
    const RCP<MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> >&solver,
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
    Require( !d_solver.is_null() );

    if ( !d_plist.is_null() )
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
RCP<const LinearOpSourceBase<Scalar> > MCLSLinearOpWithSolve::extract_fwdOpSrc()
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
RCP<const LinearOpSourceBase<Scalar> > MCLSLinearOpWithSolve::extract_prec()
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
boll MCLSLinearOpWithSolve::isExternalPrec() const
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
MCLSLinearOpWithSolve::extract_approxFwdOpSrc()
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
ESupportSolveUse MCLSLinearOpWithSolve::supportSolveUse()
{
    return d_support_solve_use;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Uninitializes and returns stored quantities.
 */
template<class Scalar>
void MCLSLinearOpWithSolve::uninitialize(
    RCP<MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> >& linear_problem,
    RCP<Teuchos::ParameterList>& plist,
    RCP<MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> >&solver,
    RCP<const LinearOpSourceBase<Scalar> >& fwd_op_src,
    RCP<const PreconditionerBase<Scalar> >& prec,
    bool& is_external_prec,
    <const LinearOpSourceBase<Scalar> >& approx_fwd_op_src,
    ESupportSolveUse& support_solve_use )
{
    if ( !linear_problem.is_null() ) linear_problem = d_linear_problem;
    if ( !plist.is_null() ) plist = d_plist;
    if ( !solver.is_null() ) solver = d_solver;
    if ( !fwd_op_src.is_null() ) fwd_op_src = d_fwd_op_src;
    if ( !prec.is_null() ) prec = d_prec;
    is_external_prec = d_external_prec;
    if ( !approx_fwd_op_src.is_null() ) approx_fwd_op_src = d_approx_fwd_op_src;
    support_solve_use = d_support_solve_use;

    d_linear_problem = Teuchos::null;
    d_plist = Teuchos::null;
    d_solver = Teuchos::null;
    d_fwd_op_src = Teuchos::null;
    d_prec = Teuchos::null;
    d_is_external_prec = false;
    d_approx_fwd_src = Teuchos::null;
    d_support_solve_use = SUPPORT_SOLVE_UNSPECIFIED;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get the range of the operator for the linear problem being solved by
 * this solver.
 */
template<class Scalar>
RCP<const VectorSpaceBase<Scalar> > MCLSLinearOpWithSolve::range() const
{
    if ( !d_linear_problem.is_null() ) 
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
RCP<const VectorSpaceBase<Scalar> > MCLSLinearOpWithSolve::domain() const
{
    if ( !d_linear_problem.is_null() ) 
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
RCP<const VectorSpaceBase<Scalar> > MCLSLinearOpWithSolve::clone() const
{
    // Not supported.
    return Teuchos::null;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether a given operator transpose type is supported.
 */
template<class Scalar>
bool MCLSLinearOpWithSolve::opSupportedImpl( const EOpTransp M_trans )
{
    return ::Thyra::opSupported( d_linear_problem->getOperator(), M_trans );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Apply the linear operator from the linear problem being solved by
 * this solver to a multivector.
 */
template<class Scalar>
void MCLSLinearOpWithSolve::applyImpl( const EOpTransp M_trans,
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
bool MCLSLinearOpWithSolve::solveSupportsImpl( const EOpTransp M_trans )
{
    return solveSupportsNewImpl( M_trans, Teuchos::null );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether this solver supports the given transpose type and
 * solve criteria.
 */
template<class Scalar>
bool MCLSLinearOpWithSolve::solveSupportsNewImpl( 
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

    return false;;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Return whether this solver supports the given transpose type and
    solve measure type. 
*/
template<class Scalar>
bool MCLSLinearOpWithSolve::solveSupportsSolveMeasureTypeImpl(
    EOpTransp M_trans, const SolveMeasureType& solveMeasureType ) const
{
  SolveCriteria<Scalar> solveCriteria(
      solveMeasureType, SolveCriteria<Scalar>::unspecifiedTolerance() );
  return solveSupportsNewImpl( M_trans, Teuchos::constOptInArg(solveCriteria) );
}

//---------------------------------------------------------------------------//

} // end namespace Thyra

//---------------------------------------------------------------------------//

#endif // end THYRA_MCLS_LINEAR_OP_WITH_SOLVE_IMPL_HPP

//---------------------------------------------------------------------------//
// end Thyra_MCLSLinearOpWithSolve.hpp
//---------------------------------------------------------------------------//

