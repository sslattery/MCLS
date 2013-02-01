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

#include "MCLS_SolverManager.hpp"
#include "Thyra_MCLSLinearProblemAdapter.hpp"
#include "Thyra_MCLSSolverManagerAdapter.hpp"

#include <Thyra_LinearOpWithSolveBase.hpp>
#include <Thyra_LinearOpSourceBase.hpp>

namespace Thyra
{

//---------------------------------------------------------------------------//
/*!
 * \class SolverManager
 * \brief Linear solver base class.
 */
template<class Scalar>
class MCLSLinearOpWithSolve : virtual public LinearOpWithSolveBase<Scalar>
{
  public:

    //@{
    //! Typedefs.
    typedef MultiVectorBase<Scalar>                  MultiVector;
    typedef LinearOpBase<Scalar>                     LinearOp;
    //@}

    // Uninitialized constructor.
    MCLSLinearOpWithSolve();

    /** \brief Initializes given precreated solver objects. */
    void initialize(
	const RCP<MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> >& linear_problem,
	const RCP<Teuchos::ParameterList>& plist,
	const RCP<MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> >&solver,
	const RCP<const LinearOpSourceBase<Scalar> >& fwd_op_src,
	const RCP<const PreconditionerBase<Scalar> >& prec,
	const bool is_external_prec,
	const RCP<const LinearOpSourceBase<Scalar> >& approx_fwd_op_src,
	const ESupportSolveUse& support_solve_use,
	const int conv_check_freq );
  

    /** @name Extraction methods */
    //@{
    /** \brief Extract the forward <tt>LinearOpBase<double></tt> object so that
     * it can be modified.
     */
    RCP<const LinearOpSourceBase<double> > extract_fwdOpSrc();

    /** \brief Extract the preconditioner.
     */
    RCP<const PreconditionerBase<double> > extract_prec();

    /** \brief Determine if the preconditioner was external or not.
     */
    bool isExternalPrec() const;

    /** \brief Extract the approximate forward <tt>LinearOpBase<double></tt>
     * object used to build the preconditioner.
     */
    RCP<const LinearOpSourceBase<double> > extract_approxFwdOpSrc();
    //@}


    /** @name Overridden from LinearOpBase */
    //@{
    /** \brief. */
    RCP<const VectorSpaceBase<double> > range() const;

    /** \brief. */
    RCP<const VectorSpaceBase<double> > domain() const;

    /** \brief. */
    RCP<const LinearOpBase<double> > clone() const;
    //@}

  protected: 

    /** @name Overridden from LinearOpBase  */
    //@{
    /** \brief . */
    virtual bool opSupportedImpl( EOpTransp M_trans ) const;

    /** \brief . */
    virtual void applyImpl( const EOpTransp M_trans,
			    const MultiVectorBase<double> &X,
			    const Ptr<MultiVectorBase<double> > &Y,
			    const double alpha,
			    const double beta ) const;
    //@}


    /** @name Overridden from LinearOpWithSolveBase. */
    //@{
    /** \brief . */
    virtual bool solveSupportsImpl( EOpTransp M_trans ) const;

    /** \brief . */
    virtual bool solveSupportsNewImpl(
	EOpTransp transp,
	const Ptr<const SolveCriteria<Scalar> > solveCriteria ) const;

    /** \brief . */
    virtual bool solveSupportsSolveMeasureTypeImpl(
	EOpTransp M_trans, const SolveMeasureType& solveMeasureType ) const;

    /** \brief . */
    virtual SolveStatus<Scalar> solveImpl(
	const EOpTransp transp,
	const MultiVectorBase<Scalar> &B,
	const Ptr<MultiVectorBase<Scalar> > &X,
	const Ptr<const SolveCriteria<Scalar> > solveCriteria ) const;
    //@}

  private:

    // Check the initialization.
    void assertInitialized() const;

  private:

    // Blocked linear problem.
    RCP<MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> > d_linear_problem;

    // Solver parameters.
    RCP<Teuchos::ParameterList> d_plist;

    // Blocked solver manager.
    RCP<MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> > d_solver;

    // Check frequency for convergence.
    int d_conv_check_freq;

    // Linear operator source.
    RCP<const LinearOpSourceBase<Scalar> > d_fwd_op_src;

    // Preconditioner.
    RCP<const PreconditionerBase<Scalar> > d_prec;

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

#endif // end THYRA_MCLS_LINEAR_OP_WITH_SOLVE_HPP

//---------------------------------------------------------------------------//
// end Thyra_MCLSLinearOpWithSolve.hpp
//---------------------------------------------------------------------------//

