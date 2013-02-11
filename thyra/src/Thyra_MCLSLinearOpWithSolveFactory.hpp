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

#ifndef THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_HPP
#define THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_HPP

#include <Teuchos_ParameterList.hpp>

#include <Epetra_RowMatrix.h>

#include <Tpetra_CrsMatrix.hpp>

#include <Thyra_LinearOpWithSolveFactoryBase.hpp>

namespace Thyra {

//---------------------------------------------------------------------------//
enum EMCLSSolverType {
    SOLVER_TYPE_MCSA,
    SOLVER_TYPE_SEQUENTIAL_MC,
    SOLVER_TYPE_ADJOINT_MC
};

inline std::istream& operator>>(
    std::istream& is, EMCLSSolverType& sType)
{
    int intval;
    is >> intval;
    sType = (EMCLSSolverType)intval;
    return is;
}


//---------------------------------------------------------------------------//
/** \brief <tt>LinearOpWithSolveFactoryBase</tt> subclass implemented in terms
 * of <tt>MCLS</tt>.
 *
 * \ingroup MCLS_Thyra_adapters_grp
 */
template<class Scalar>
class MCLSLinearOpWithSolveFactory : public LinearOpWithSolveFactoryBase<Scalar> {
  public:

    /** \name Public types */
    //@{
    /** \brief . */

    typedef typename Teuchos::ScalarTraits<Scalar>::magnitudeType  MagnitudeType;

    //@}

    /** \name Parameter names for Parameter List */
    //@{

    /** \brief . */
    static const std::string  SolverType_name;
    /** \brief . */           
    static const std::string  SolverType_default;
    /** \brief . */
    static const std::string  SolverTypes_name;
    /** \brief . */
    static const std::string  MCSA_name;
    /** \brief . */
    static const std::string  SequentialMC_name;
    /** \brief . */
    static const std::string  AdjointMC_name;
    /** \brief . */
    static const std::string  ConvergenceTestFrequency_name;

    //@}

    /** @name Constructors/initializers/accessors */
    //@{

    /** \brief Construct without preconditioner factory. */
    MCLSLinearOpWithSolveFactory();

    /** \brief Calls <tt>this->setPreconditionerFactory(precFactory)</tt.  . */
    MCLSLinearOpWithSolveFactory(
	const Teuchos::RCP<PreconditionerFactoryBase<Scalar> > &precFactory );

    //@}

    /** @name Overridden public functions from LinearOpWithSolveFactoryBase */
    //@{
    /** \brief  . */
    bool acceptsPreconditionerFactory() const;

    /** \brief . */
    void setPreconditionerFactory(
	const Teuchos::RCP<PreconditionerFactoryBase<Scalar> >& precFactory,
	const std::string& precFactoryName
	);

    /** \brief . */
    Teuchos::RCP<PreconditionerFactoryBase<Scalar> > getPreconditionerFactory() const;

    /** \brief . */
    void unsetPreconditionerFactory(
	Teuchos::RCP<PreconditionerFactoryBase<Scalar> >* precFactory,
	std::string* precFactoryName
	);

    /** \brief . */
    bool isCompatible( const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;

    /** \brief . */
    Teuchos::RCP<LinearOpWithSolveBase<Scalar> > createOp() const;

    /** \brief . */
    void initializeOp(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
	LinearOpWithSolveBase<Scalar>* Op,
	const ESupportSolveUse supportSolveUse ) const;

    /** \brief . */
    void initializeAndReuseOp(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
	LinearOpWithSolveBase<Scalar>* Op ) const;

    /** \brief . */
    void uninitializeOp(
	LinearOpWithSolveBase<Scalar>* Op,
	Teuchos::RCP<const LinearOpSourceBase<Scalar> >* fwdOpSrc,
	Teuchos::RCP<const PreconditionerBase<Scalar> >* prec,
	Teuchos::RCP<const LinearOpSourceBase<Scalar> >* approxFwdOpSrc,
	ESupportSolveUse* supportSolveUse ) const;

    /** \brief . */
    bool supportsPreconditionerInputType(
	const EPreconditionerInputType precOpType) const;

    /** \brief . */
    void initializePreconditionedOp(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
	const Teuchos::RCP<const PreconditionerBase<Scalar> >& prec,
	LinearOpWithSolveBase<Scalar>* Op,
	const ESupportSolveUse supportSolveUse ) const;

    /** \brief . */
    void initializeApproxPreconditionedOp(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
	LinearOpWithSolveBase<Scalar>* Op,
	const ESupportSolveUse supportSolveUse ) const;
    //@}

    /** @name Overridden from ParameterListAcceptor */
    //@{

    /** \brief . */
    void setParameterList(Teuchos::RCP<Teuchos::ParameterList> const& paramList);

    /** \brief . */
    Teuchos::RCP<Teuchos::ParameterList> getNonconstParameterList();

    /** \brief . */
    Teuchos::RCP<Teuchos::ParameterList> unsetParameterList();

    /** \brief . */
    Teuchos::RCP<const Teuchos::ParameterList> getParameterList() const;

    /** \brief . */
    Teuchos::RCP<const Teuchos::ParameterList> getValidParameters() const;

    //@}

    /** \name Public functions overridden from Teuchos::Describable. */
    //@{

    /** \brief . */
    std::string description() const;

    //@}

  private:

    // Generate valid parameters for this factory.
    static Teuchos::RCP<const Teuchos::ParameterList> generateAndGetValidParameters();

    // Update the valid parameter list for this factory.
    void updateThisValidParamList();

    // Select the appropriate subclasses to initialize the linear solver with.
    void selectOpImpl(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
	const Teuchos::RCP<const PreconditionerBase<Scalar> >& prec,
	const bool reusePrec,
	LinearOpWithSolveBase<Scalar>* Op,
	const ESupportSolveUse supportSolveUse ) const;

    // Initialize the linear operator.
    template<class MultiVector, class Matrix>
    void initializeOpImpl(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
	const Teuchos::RCP<const PreconditionerBase<Scalar> >& prec,
	const bool reusePrec,
	LinearOpWithSolveBase<Scalar>* Op,
	const ESupportSolveUse supportSolveUse ) const;

    // Get an Epetra_RowMatrix from the linear operator source.
    Teuchos::RCP<const Epetra_RowMatrix> getEpetraRowMatrix(
	const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;

    // Check for Epetra compatiblity.
    bool isEpetraCompatible( const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;

    // Get a Tpetra::CrsMatrix from the linear operator source.
    template<class LO, class GO>
    Teuchos::RCP<const Tpetra::CrsMatrix<Scalar,LO,GO> >
    getTpetraCrsMatrix( const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;

    // Check for Tpetra compatiblity.
    bool isTpetraCompatible( const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;

  private:

    // Preconditioner factory.
    Teuchos::RCP<PreconditionerFactoryBase<Scalar> >  d_prec_factory;

    // Precondition factory name.
    std::string d_prec_factory_name;

    // Valid parameter list.
    Teuchos::RCP<Teuchos::ParameterList> d_valid_plist;

    // Parameter list.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

    // Solver type.
    EMCLSSolverType d_solver_type;

    // Convergence check frequency.
    int d_convergence_test_frequency;
};

//@}
//---------------------------------------------------------------------------//

} // end namespace Thyra

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "Thyra_MCLSLinearOpWithSolveFactory_impl.hpp"

//---------------------------------------------------------------------------//

#endif // THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_HPP

//---------------------------------------------------------------------------//
// end ThyraMCLSLinearOpWithSolveFactory.hpp
//---------------------------------------------------------------------------//

