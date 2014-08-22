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
 * \file MCLS_MCSAModelEvaluator.hpp
 * \author Stuart R. Slattery
 * \brief Monte Carlo Synthetic Acceleration solver manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MCSASOLVERMANAGER_HPP
#define MCLS_MCSASOLVERMANAGER_HPP

#include <random>

#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_Xorshift.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_ScalarTraits.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_Comm.hpp>

#include "Thyra_StateFuncModelEvaluatorBase.hpp"

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MCSAModelEvaluator
 * \brief Solver manager for Monte Carlo synthetic acceleration.
 */
template<class Vector, class Matrix, class RNG = Xorshift<> >
class MCSAModelEvaluator : 
	public ::Thyra::StateFuncModelEvaluatorBase<typename VectorTraits<Vector>::scalar_type>
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                       vector_type;
    typedef VectorTraits<vector_type>                    VT;
    typedef typename VT::scalar_type                     Scalar;
    typedef ::Thyra::StateFuncModelEvaluatorBase<Scalar> Base;
    typedef Matrix                                       matrix_type;
    typedef MatrixTraits<vector_type,matrix_type>        MT;
    typedef LinearProblem<vector_type,matrix_type>       LinearProblemType;
    typedef Teuchos::Comm<int>                           Comm;
    //@}

    // Default constructor.
    MCSAModelEvaluator( const Teuchos::RCP<const Comm>& global_comm,
			const Teuchos::RCP<Teuchos::ParameterList>& plist );

    // Constructor.
    MCSAModelEvaluator( const Teuchos::RCP<const Comm>& global_comm,
			const Teuchos::RCP<Teuchos::ParameterList>& plist,
			const Teuchos::RCP<const matrix_type>& A,
			const Teuchos::RCP<const vector_type>& b,
			const Teuchos::RCP<const matrix_type>& M = Teuchos::null );

    //! Destructor.
    ~MCSAModelEvaluator() { /* ... */ }

    // Set the linear problem with the manager.
    void setProblem( const Teuchos::RCP<const matrix_type>& A,
		     const Teuchos::RCP<const vector_type>& b,
		     const Teuchos::RCP<const matrix_type>& M = Teuchos::null );

    // Set the parameters for the manager. The manager will modify this list
    // with default parameters that are not defined.
    void setParameters( const Teuchos::RCP<Teuchos::ParameterList>& params );

    // Get the preconditioned residual given a LHS.
    Teuchos::RCP<vector_type> 
    getPrecResidual( const Teuchos::RCP<vector_type>& x ) const;

    // Get the RHS.
    Teuchos::RCP<vector_type> getRHS() const
    { return d_b; }

  public:

    // Public functions overridden from ModelEvaulator.
    Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_x_space() const;
    Teuchos::RCP<const ::Thyra::VectorSpaceBase<Scalar> > get_f_space() const;
    ::Thyra::ModelEvaluatorBase::InArgs<Scalar> getNominalValues() const;
    ::Thyra::ModelEvaluatorBase::InArgs<Scalar> createInArgs() const;

  private:

    // Private functions overridden from ModelEvaulatorDefaultBase
    ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> createOutArgsImpl() const;

    void evalModelImpl(
      const ::Thyra::ModelEvaluatorBase::InArgs<Scalar> &inArgs,
      const ::Thyra::ModelEvaluatorBase::OutArgs<Scalar> &outArgs ) const;

  private:

    Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > d_x_space;
    Teuchos::RCP<const Thyra::VectorSpaceBase<Scalar> > d_f_space;

  private:

    // Build the preconditioned residual given the current state.
    Teuchos::RCP<vector_type> getPrecResidual( 
	const Teuchos::RCP<vector_type>& x ) const;

    // Build the residual Monte Carlo problem from the input problem.
    void buildResidualMonteCarloProblem();

  private:

    // Linear operator.
    Teuchos::RCP<const matrix_type> d_A;

    // Right-hand side.
    Teuchos::RCP<const vector_type> d_b;

    // Preconditioner.
    Teuchos::RCP<const matrix_type> d_M;

    // Residual
    Teuchos::RCP<vector_type> d_r;

    // Residual linear problem
    Teuchos::RCP<LinearProblemType> d_mc_problem;

    // Global communicator.
    Teuchos::RCP<const Comm> d_global_comm;

    // Parameters.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Monte Carlo solver manager.
    Teuchos::RCP<SolverManager<vector_type,matrix_type> > d_mc_solver;

    // Fixed point iteration.
    Teuchos::RCP<FixedPointType> d_fixed_point;

    // Number of smoothing steps.
    int d_num_smooth;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_MCSAModelEvaluator_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MCSAMODELEVALUATOR_HPP

//---------------------------------------------------------------------------//
// end MCLS_MCSAModelEvaluator.hpp
//---------------------------------------------------------------------------//

