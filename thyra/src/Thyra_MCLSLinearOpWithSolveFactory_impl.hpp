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
 * \file Thyra_MCLSLinearOpWithSolveFactory_impl.hpp
 * \author Stuart R. Slattery
 * \brief Thyra LinearOpWithSolve implementation for MCLS.
 */
//---------------------------------------------------------------------------//

#ifndef THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_IMPL_HPP
#define THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_IMPL_HPP

#include <Thyra_MCLSLinearOpWithSolve.hpp>

#include <MCLS_DBC.hpp>
#include <MCLS_SolverManager.hpp>
#include <MCLS_MCSASolverManager.hpp>
#include <MCLS_SequentialMCSolverManager.hpp>
#include <MCLS_AdjointSolverManager.hpp>
#include <MCLS_ForwardSolverManager.hpp>
#include <MCLS_FixedPointSolverManager.hpp>
#include <MCLS_MatrixTraits.hpp>

#include "MCLS_LinearProblemAdapter.hpp"
#include "MCLS_SolverManagerAdapter.hpp"

#include <Teuchos_VerboseObjectParameterListHelpers.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_dyn_cast.hpp>
#include <Teuchos_ValidatorXMLConverterDB.hpp>
#include <Teuchos_StandardValidatorXMLConverters.hpp>
#include <Teuchos_as.hpp>
#include <Teuchos_DefaultComm.hpp>

#include <Thyra_EpetraOperatorViewExtractorStd.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

#include <Epetra_Operator.h>
#include <Epetra_RowMatrix.h>

#include <Tpetra_Operator.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace Thyra {


//---------------------------------------------------------------------------//
// Parameter names for Parameter List
template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::SolverType_name = 
    "Solver Type";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::SolverType_default = 
    "MCSA";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::SolverTypes_name = 
    "Solver Types";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::MCSA_name = "MCSA";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::SequentialMC_name = 
    "Sequential MC";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::AdjointMC_name = 
    "Adjoint MC";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::ForwardMC_name = 
    "Forward MC";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::FixedPoint_name = 
    "Fixed Point";

template<class Scalar>
const std::string MCLSLinearOpWithSolveFactory<Scalar>::ConvergenceTestFrequency_name 
= "Convergence Test Frequency";

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
template<class Scalar>
MCLSLinearOpWithSolveFactory<Scalar>::MCLSLinearOpWithSolveFactory()
    : d_solver_type( SOLVER_TYPE_MCSA )
    , d_convergence_test_frequency( 1 )
{
    updateThisValidParamList();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Preconditioner factor constructor.
 */
template<class Scalar>
MCLSLinearOpWithSolveFactory<Scalar>::MCLSLinearOpWithSolveFactory(
    const RCP<PreconditionerFactoryBase<Scalar> >& precFactory )
    : d_solver_type( SOLVER_TYPE_MCSA )
{
    this->setPreconditionerFactory(precFactory, "");
}


//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
bool MCLSLinearOpWithSolveFactory<Scalar>::acceptsPreconditionerFactory() const
 {
     return true;
 }


//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::setPreconditionerFactory(
    const RCP<PreconditionerFactoryBase<Scalar> >& precFactory,
    const std::string& precFactoryName )
{
    MCLS_REQUIRE( Teuchos::nonnull(precFactory) );

    RCP<const Teuchos::ParameterList> precFactory_valid_plist = 
	precFactory->getValidParameters();
    const std::string d_precFactoryName =
	( precFactoryName != ""
	  ? precFactoryName
	  : ( precFactory_valid_plist.get() ? 
	      precFactory_valid_plist->name() : 
	      "GENERIC PRECONDITIONER FACTORY" ) );

    d_prec_factory = precFactory;
    d_prec_factory_name = precFactoryName;
    updateThisValidParamList();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
RCP<PreconditionerFactoryBase<Scalar> >
MCLSLinearOpWithSolveFactory<Scalar>::getPreconditionerFactory() const
{
    return d_prec_factory;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::unsetPreconditionerFactory(
    RCP<PreconditionerFactoryBase<Scalar> >* precFactory,
    std::string* precFactoryName )
{
    if(precFactory) *precFactory = d_prec_factory;
    if(precFactoryName) *precFactoryName = d_prec_factory_name;
    d_prec_factory = Teuchos::null;
    d_prec_factory_name = "";
    updateThisValidParamList();
}


//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
bool MCLSLinearOpWithSolveFactory<Scalar>::isCompatible(
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    // Check the preconditioner factory for compatibility. 
    bool prec_compatible = true;
    if( Teuchos::nonnull(d_prec_factory) )
    {
	prec_compatible = d_prec_factory->isCompatible(fwdOpSrc);
    }

    // MCLS interfaces are currently only implemented for Epetra_RowMatrix and
    // Tpetra::CrsMatrix. 
    bool epetra_compatible = isEpetraCompatible( fwdOpSrc );
    bool tpetra_compatible = ( isTpetraCompatible<int,int>(fwdOpSrc) ||
			       isTpetraCompatible<int,long>(fwdOpSrc) );
    bool solve_compatible = ( epetra_compatible || tpetra_compatible );

    return ( prec_compatible && solve_compatible );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
RCP<LinearOpWithSolveBase<Scalar> >
MCLSLinearOpWithSolveFactory<Scalar>::createOp() const
{
    return Teuchos::rcp(new MCLSLinearOpWithSolve<Scalar>());
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::initializeOp(
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    LinearOpWithSolveBase<Scalar>* Op,
    const ESupportSolveUse supportSolveUse ) const
{
    using Teuchos::null;
    selectOpImpl(fwdOpSrc,null,null,false,Op,supportSolveUse);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::initializeAndReuseOp(
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    LinearOpWithSolveBase<Scalar>* Op ) const
{
    using Teuchos::null;
    selectOpImpl(fwdOpSrc,null,null,true,Op,SUPPORT_SOLVE_UNSPECIFIED);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
bool MCLSLinearOpWithSolveFactory<Scalar>::supportsPreconditionerInputType(
    const EPreconditionerInputType precOpType
    ) const
{
    if( Teuchos::nonnull(d_prec_factory) ) return true;
    return (precOpType==PRECONDITIONER_INPUT_TYPE_AS_OPERATOR);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::initializePreconditionedOp(
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    const RCP<const PreconditionerBase<Scalar> >& prec,
    LinearOpWithSolveBase<Scalar>* Op,
    const ESupportSolveUse supportSolveUse ) const
{
    using Teuchos::null;
    selectOpImpl(fwdOpSrc,null,prec,false,Op,supportSolveUse);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::initializeApproxPreconditionedOp(
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    const RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
    LinearOpWithSolveBase<Scalar>* Op,
    const ESupportSolveUse supportSolveUse ) const
{
    using Teuchos::null;
    selectOpImpl(fwdOpSrc,approxFwdOpSrc,null,false,Op,supportSolveUse);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from LinearOpWithSolveFactoryBase
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::uninitializeOp(
    LinearOpWithSolveBase<Scalar>* Op,
    RCP<const LinearOpSourceBase<Scalar> >* fwdOpSrc,
    RCP<const PreconditionerBase<Scalar> >* prec,
    RCP<const LinearOpSourceBase<Scalar> >* approxFwdOpSrc,
    ESupportSolveUse* supportSolveUse ) const
{
    MCLS_REQUIRE( Op != NULL );

    MCLSLinearOpWithSolve<Scalar>& mclsOp = 
	Teuchos::dyn_cast<MCLSLinearOpWithSolve<Scalar> >(*Op);
    RCP<const LinearOpSourceBase<Scalar> > _fwdOpSrc = 
	mclsOp.extract_fwdOpSrc();
    RCP<const PreconditionerBase<Scalar> > _prec = 
	( mclsOp.isExternalPrec() ? mclsOp.extract_prec() : Teuchos::null );

    // Note: above we only extract the preconditioner if it was passed in
    // externally.  Otherwise, we need to hold on to it so that we can reuse it
    // in the next initialization.
    RCP<const LinearOpSourceBase<Scalar> > _approxFwdOpSrc = 
	mclsOp.extract_approxFwdOpSrc();

    ESupportSolveUse _supportSolveUse = 
	mclsOp.supportSolveUse();

    if(fwdOpSrc) *fwdOpSrc = _fwdOpSrc;
    if(prec) *prec = _prec;
    if(approxFwdOpSrc) *approxFwdOpSrc = _approxFwdOpSrc;
    if(supportSolveUse) *supportSolveUse = _supportSolveUse;
}


//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from ParameterListAcceptor
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::setParameterList(
    const RCP<Teuchos::ParameterList>& paramList )
{
    MCLS_REQUIRE( Teuchos::nonnull(paramList) );

    paramList->validateParametersAndSetDefaults(*this->getValidParameters(), 1);
    d_plist = paramList;
    d_solver_type =
	Teuchos::getIntegralValue<EMCLSSolverType>(*d_plist, SolverType_name);
    d_convergence_test_frequency =
	Teuchos::getParameter<int>(*d_plist, ConvergenceTestFrequency_name);
    Teuchos::readVerboseObjectSublist(&*d_plist,this);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from ParameterListAcceptor
 */
template<class Scalar>
RCP<Teuchos::ParameterList>
MCLSLinearOpWithSolveFactory<Scalar>::getNonconstParameterList()
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from ParameterListAcceptor
 */
template<class Scalar>
RCP<Teuchos::ParameterList>
MCLSLinearOpWithSolveFactory<Scalar>::unsetParameterList()
{
    RCP<Teuchos::ParameterList> _paramList = d_plist;
    d_plist = Teuchos::null;
    return _paramList;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from ParameterListAcceptor
 */
template<class Scalar>
RCP<const Teuchos::ParameterList>
MCLSLinearOpWithSolveFactory<Scalar>::getParameterList() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from ParameterListAcceptor
 */
template<class Scalar>
RCP<const Teuchos::ParameterList>
MCLSLinearOpWithSolveFactory<Scalar>::getValidParameters() const
{
    return d_valid_plist;
}


//---------------------------------------------------------------------------//
/*!
 * \brief Overridden from Teuchos::Describable
 */
template<class Scalar>
std::string MCLSLinearOpWithSolveFactory<Scalar>::description() const
{
    std::ostringstream oss;
    oss << "Thyra::MCLSLinearOpWithSolveFactory";
    return oss.str();
}

//---------------------------------------------------------------------------//
/*!
 * \brief Generate valid parameters for this factory.
 */
template<class Scalar>
RCP<const Teuchos::ParameterList>
MCLSLinearOpWithSolveFactory<Scalar>::generateAndGetValidParameters()
{
    using Teuchos::as;
    using Teuchos::tuple;
    using Teuchos::setStringToIntegralParameter;
    Teuchos::ValidatorXMLConverterDB::addConverter(
	Teuchos::DummyObjectGetter<
	Teuchos::StringToIntegralParameterEntryValidator<EMCLSSolverType> 
	>::getDummyObject(),
	Teuchos::DummyObjectGetter<Teuchos::StringToIntegralValidatorXMLConverter<
	EMCLSSolverType> >::getDummyObject());

    static RCP<Teuchos::ParameterList> validParamList;
    if( validParamList.get()==NULL ) 
    {
	validParamList = Teuchos::rcp(new Teuchos::ParameterList(
					  "MCLSLinearOpWithSolveFactory") );

	setStringToIntegralParameter<EMCLSSolverType>(
	    SolverType_name, SolverType_default,
	    "Type of linear solver algorithm to use.",
	    tuple<std::string>(
		"MCSA",
		"Sequential MC",
		"Adjoint MC",
		"Forward MC",
                "Fixed Point"
		),
	    tuple<std::string>(
		"Monte Carlo Synthetic Acceleration solver for nonsymmetric linear "
		"systems that performs single right-hand side solves on multiple "
		"right-hand sides sequentially.",

		"Sequential Monte Carlo solver for nonsymmetric linear systems "
		"that performs single right-hand side solves on multiple "
		"right-hand sides sequentially.",

		"Adjoint Monte Carlo solver for nonsymmetric linear systems "
		"that performs single right-hand side solves on multiple "
		"right-hand sides sequentially.",

		"Forward Monte Carlo solver for nonsymmetric linear systems "
		"that performs single right-hand side solves on multiple "
		"right-hand sides sequentially.",

                "Fixed point iteration. Iteration type determined by "
                "'Fixed Point Type' parameter"
		),
	    tuple<EMCLSSolverType>(
		SOLVER_TYPE_MCSA,
		SOLVER_TYPE_SEQUENTIAL_MC,
		SOLVER_TYPE_ADJOINT_MC,
		SOLVER_TYPE_FORWARD_MC,
                SOLVER_TYPE_FIXED_POINT
		),
	    &*validParamList
	    );
	validParamList->set(
	    ConvergenceTestFrequency_name, as<int>(1),
	    "Number of linear solver iterations to skip between applying"
	    " user-defined convergence test.");

	// We'll use Epetra to get the valid parameters for the solver
	// subclasses as they are independent of the vector/operator
	// implementation used.
	Teuchos::ParameterList
	    &solverTypesSL = validParamList->sublist(SolverTypes_name);
	{
	    MCLS::MCSASolverManager<Epetra_Vector,Epetra_RowMatrix> mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(MCSA_name).setParameters(
		*(mgr.getValidParameters()) );

	    // We need an MC solver manager here as well to get those
	    // parameters as we can't gather them from the solver manager
	    // until the linear problem is set.
	    MCLS::AdjointSolverManager<Epetra_Vector,Epetra_RowMatrix> mc_mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(MCSA_name).setParameters(
		*(mc_mgr.getValidParameters()) );
	}
	{
	    MCLS::SequentialMCSolverManager<Epetra_Vector,Epetra_RowMatrix> mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(SequentialMC_name).setParameters(
		*(mgr.getValidParameters()) );

	    // We need an MC solver manager here as well to get those
	    // parameters as we can't gather them from the solver manager
	    // until the linear problem is set.
	    MCLS::AdjointSolverManager<Epetra_Vector,Epetra_RowMatrix> mc_mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(MCSA_name).setParameters(
		*(mc_mgr.getValidParameters()) );
	}
	{
	    MCLS::AdjointSolverManager<Epetra_Vector,Epetra_RowMatrix> mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(AdjointMC_name).setParameters(
		*(mgr.getValidParameters()) );
	}
	{
	    MCLS::ForwardSolverManager<Epetra_Vector,Epetra_RowMatrix> mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(ForwardMC_name).setParameters(
		*(mgr.getValidParameters()) );
	}
	{
	    MCLS::FixedPointSolverManager<Epetra_Vector,Epetra_RowMatrix> mgr(
		Teuchos::DefaultComm<int>::getComm(), Teuchos::parameterList() );
	    solverTypesSL.sublist(FixedPoint_name).setParameters(
		*(mgr.getValidParameters()) );
	}
    }

    return validParamList;
}


//---------------------------------------------------------------------------//
/*!
 * \brief Update valid parameters for this factory.
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::updateThisValidParamList()
{
    d_valid_plist = Teuchos::rcp(
	new Teuchos::ParameterList(*generateAndGetValidParameters()) );
    Teuchos::setupVerboseObjectSublist(&*d_valid_plist);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Select the appropriate subclasses to initialize the linear solver
 * with. 
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::selectOpImpl(
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    const RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
    const RCP<const PreconditionerBase<Scalar> >& prec_in,
    const bool reusePrec,
    LinearOpWithSolveBase<Scalar>* Op,
    const ESupportSolveUse supportSolveUse ) const
{
    // Get the MCLSLinearOpWithSolve interface
    MCLSLinearOpWithSolve<Scalar>
	*mclsOp = &Teuchos::dyn_cast<MCLSLinearOpWithSolve<Scalar> >(*Op);

    // Get the approximate forward operator.
    RCP<const LinearOpBase<Scalar> > approxFwdOp = 
	( approxFwdOpSrc.get() ? approxFwdOpSrc->getOp() : Teuchos::null );

    // Get/Create the preconditioner
    RCP<PreconditionerBase<Scalar> > my_prec = Teuchos::null;
    RCP<const PreconditionerBase<Scalar> > prec = Teuchos::null;
    if( prec_in.get() ) 
    {
	// Use an externally defined preconditioner
	prec = prec_in;
    }
    else 
    {
	// Try and generate a preconditioner on our own
	if( d_prec_factory.get() ) 
	{
	    my_prec = ( 
		!mclsOp->isExternalPrec()
		? Teuchos::rcp_const_cast<PreconditionerBase<Scalar> >(
		    mclsOp->extract_prec()) : Teuchos::null );
	    bool hasExistingPrec = false;
	    if( my_prec.get() ) 
	    {
		hasExistingPrec = true;
		// ToDo: Get the forward operator and validate that it is the
		// same operator that is used here!
	    }
	    else 
	    {
		hasExistingPrec = false;
		my_prec = d_prec_factory->createPrec();
	    }
	    if( hasExistingPrec && reusePrec ) 
	    {
		// Just reuse the existing preconditioner again!
	    }
	    else 
	    {
		// Update the preconditioner
		if( approxFwdOp.get() )
		{
		    d_prec_factory->initializePrec(approxFwdOpSrc,&*my_prec);
		}
		else
		{
		    d_prec_factory->initializePrec(fwdOpSrc,&*my_prec);
		}
	    }
	    prec = my_prec;
	}
    }

    // Set the preconditioner. Currently we do not handle unspecified
    // preconditioners.
    Teuchos::RCP<const LinearOpBase<Scalar> > left;
    Teuchos::RCP<const LinearOpBase<Scalar> > right;
    if( prec.get() ) 
    {
	left = prec->getLeftPrecOp();
	right = prec->getRightPrecOp();
    }

    // Initialize with the appropriate subclass.
    if ( isEpetraCompatible(*fwdOpSrc) )
    {
  	Teuchos::RCP<const Epetra_RowMatrix> left_prec;
	if ( Teuchos::nonnull(left) )
	{
	    EpetraOperatorViewExtractorStd epetraFwdOpViewExtractor;
	    RCP<const Epetra_Operator> epetraFwdOp;
	    EOpTransp epetraFwdOpTransp;
	    EApplyEpetraOpAs epetraFwdOpApplyAs;
	    EAdjointEpetraOp epetraFwdOpAdjointSupport;
	    double epetraFwdOpScalar;
	    epetraFwdOpViewExtractor.getEpetraOpView(
		left, 
		Teuchos::outArg(epetraFwdOp), 
		Teuchos::outArg(epetraFwdOpTransp),
		Teuchos::outArg(epetraFwdOpApplyAs), 
		Teuchos::outArg(epetraFwdOpAdjointSupport),
		Teuchos::outArg(epetraFwdOpScalar)
		);

	    left_prec = Teuchos::rcp_dynamic_cast<const Epetra_RowMatrix>(epetraFwdOp);
   	}

  	Teuchos::RCP<const Epetra_RowMatrix> right_prec;
	if ( Teuchos::nonnull(right) )
	{
	    EpetraOperatorViewExtractorStd epetraFwdOpViewExtractor;
	    RCP<const Epetra_Operator> epetraFwdOp;
	    EOpTransp epetraFwdOpTransp;
	    EApplyEpetraOpAs epetraFwdOpApplyAs;
	    EAdjointEpetraOp epetraFwdOpAdjointSupport;
	    double epetraFwdOpScalar;
	    epetraFwdOpViewExtractor.getEpetraOpView(
		right, 
		Teuchos::outArg(epetraFwdOp), 
		Teuchos::outArg(epetraFwdOpTransp),
		Teuchos::outArg(epetraFwdOpApplyAs), 
		Teuchos::outArg(epetraFwdOpAdjointSupport),
		Teuchos::outArg(epetraFwdOpScalar)
		);

	    right_prec = Teuchos::rcp_dynamic_cast<const Epetra_RowMatrix>(epetraFwdOp);
   	}

	initializeOpImpl<Epetra_MultiVector,
			 Epetra_RowMatrix>(
			     getEpetraRowMatrix(*fwdOpSrc),
			     left_prec, right_prec,
			     fwdOpSrc, approxFwdOpSrc, prec, my_prec,
			     reusePrec, Op, supportSolveUse );
    }
    else if ( isTpetraCompatible<int,int>(*fwdOpSrc) )
    {
	typedef int LO;
	typedef int GO;

	Teuchos::RCP<const Tpetra::CrsMatrix<Scalar,LO,GO> > left_prec;
	if ( Teuchos::nonnull(left) )
	{
	    RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetraFwdOp =
		TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraOperator(
		    left );

	    left_prec =  Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<Scalar,LO,GO> >(
		tpetraFwdOp );
	}

	Teuchos::RCP<const Tpetra::CrsMatrix<Scalar,LO,GO> > right_prec;
	if ( Teuchos::nonnull(right) )
	{
	    RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetraFwdOp =
		TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraOperator(
		    right );

	    right_prec =  Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<Scalar,LO,GO> >(
		tpetraFwdOp );
	}

	initializeOpImpl<Tpetra::MultiVector<Scalar,LO,GO>,
			 Tpetra::CrsMatrix<Scalar,LO,GO> >(
			     getTpetraCrsMatrix<LO,GO>(*fwdOpSrc),
			     left_prec, right_prec,
			     fwdOpSrc, approxFwdOpSrc, prec, my_prec,
			     reusePrec, Op, supportSolveUse );
    }
    else if ( isTpetraCompatible<int,long>(*fwdOpSrc) )
    {
	typedef int LO;
	typedef long GO;

	Teuchos::RCP<const Tpetra::CrsMatrix<Scalar,LO,GO> > left_prec;
	if ( Teuchos::nonnull(left) )
	{
	    RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetraFwdOp =
		TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraOperator(
		    left );

	    left_prec =  Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<Scalar,LO,GO> >(
		tpetraFwdOp );
	}

	Teuchos::RCP<const Tpetra::CrsMatrix<Scalar,LO,GO> > right_prec;
	if ( Teuchos::nonnull(right) )
	{
	    RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetraFwdOp =
		TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraOperator(
		    right );

	    right_prec =  Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<Scalar,LO,GO> >(
		tpetraFwdOp );
	}

	initializeOpImpl<Tpetra::MultiVector<Scalar,LO,GO>,
			 Tpetra::CrsMatrix<Scalar,LO,GO> >(
			     getTpetraCrsMatrix<LO,GO>(*fwdOpSrc),
			     left_prec, right_prec,
			     fwdOpSrc, approxFwdOpSrc, prec, my_prec,
			     reusePrec, Op, supportSolveUse );
    }
    else
    {
	TEUCHOS_TEST_FOR_EXCEPT(true);
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Initialize the linear solver.
 */
template<class Scalar>
template<class MultiVector, class Matrix>
void MCLSLinearOpWithSolveFactory<Scalar>::initializeOpImpl(
    const RCP<const Matrix>& matrix,
    const RCP<const Matrix>& left_prec,
    const RCP<const Matrix>& right_prec,
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    const RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
    const RCP<const PreconditionerBase<Scalar> >& prec,
    const Teuchos::RCP<const PreconditionerBase<Scalar> >& my_prec,
    const bool reusePrec,
    LinearOpWithSolveBase<Scalar>* Op,
    const ESupportSolveUse supportSolveUse ) const
{
    using Teuchos::rcp;
    using Teuchos::set_extra_data;
    typedef Teuchos::ScalarTraits<Scalar> ST;
    typedef typename ST::magnitudeType ScalarMag;

    const RCP<Teuchos::FancyOStream> out = this->getOStream();
    const Teuchos::EVerbosityLevel verbLevel = this->getVerbLevel();
    Teuchos::OSTab tab(out);
    if( out.get() && 
	static_cast<int>(verbLevel) > static_cast<int>(Teuchos::VERB_LOW) )
    {
	*out << "\nEntering Thyra::MCLSLinearOpWithSolveFactory<"
	     << ST::name() << ">::initializeOpImpl(...) ...\n";
    }

    TEUCHOS_TEST_FOR_EXCEPT(Op==NULL);
    TEUCHOS_TEST_FOR_EXCEPT(fwdOpSrc.get()==NULL);
    TEUCHOS_TEST_FOR_EXCEPT(fwdOpSrc->getOp().get()==NULL);
    RCP<const LinearOpBase<Scalar> > fwdOp = fwdOpSrc->getOp();
    RCP<const LinearOpBase<Scalar> > approxFwdOp = 
	( approxFwdOpSrc.get() ? approxFwdOpSrc->getOp() : Teuchos::null );

    // Get the MCLSLinearOpWithSolve interface
    MCLSLinearOpWithSolve<Scalar>
	*mclsOp = &Teuchos::dyn_cast<MCLSLinearOpWithSolve<Scalar> >(*Op);

    // Uninitialize the current solver object
    bool oldIsExternalPrec = false;
    RCP<MCLS::LinearProblemBase<Scalar> > oldLP = Teuchos::null;
    RCP<MCLS::SolverManagerBase<Scalar> > oldIterSolver = Teuchos::null;
    RCP<const LinearOpSourceBase<Scalar> > oldFwdOpSrc = Teuchos::null;
    RCP<const LinearOpSourceBase<Scalar> > oldApproxFwdOpSrc = Teuchos::null;   
    ESupportSolveUse oldSupportSolveUse = SUPPORT_SOLVE_UNSPECIFIED;
    mclsOp->uninitialize( &oldLP, NULL, &oldIterSolver, &oldFwdOpSrc,
			  NULL, &oldIsExternalPrec, &oldApproxFwdOpSrc, 
			  &oldSupportSolveUse );

    // Create the MCLS linear problem.
    typedef MCLS::LinearProblemAdapter<MultiVector,Matrix> LP_t;
    RCP<LP_t> lp = rcp(new LP_t());

    // Set the operator (this is the concrete subclass).
    lp->setOperator( matrix );

    // Set the preconditioners.
    lp->setLeftPrec( left_prec );
    lp->setRightPrec( right_prec );

    if( my_prec.get() ) 
    {
    	set_extra_data<RCP<const PreconditionerBase<Scalar> > >(
    	    my_prec,"MCLS::InternalPrec",
    	    Teuchos::inOutArg(lp), Teuchos::POST_DESTROY, false);
    }
    else if( prec.get() ) 
    {
    	set_extra_data<RCP<const PreconditionerBase<Scalar> > >(
    	    prec,"MCLS::ExternalPrec",
    	    Teuchos::inOutArg(lp), Teuchos::POST_DESTROY, false);
    }

    // Generate the parameter list.
    typedef MCLS::SolverManagerAdapter<MultiVector,Matrix> IterativeSolver_t;
    typedef typename IterativeSolver_t::Vector Vector;
    typedef MCLS::SolverManager<Vector,Matrix> Solver_t;
    RCP<IterativeSolver_t> iterativeSolver = Teuchos::null;
    RCP<Solver_t> solver = Teuchos::null;
    RCP<Teuchos::ParameterList> solverPL = Teuchos::parameterList();
  
    // Create the linear solver. We are using the linear operator communicator
    // here!!!! This will probably be MPI_COMM_WORLD in an MPI build an
    // therefore has implications for using multiple sets with MCLS where the
    // linear problem is expected to be defined on a subset of the global
    // communicator. The issue here is that these Thyra operations are
    // potentially not occuring in the global communicator scope.
    Teuchos::RCP<const Teuchos::Comm<int> > global_comm = 
        MCLS::MatrixTraits<Vector,Matrix>::getComm( *matrix );
    switch(d_solver_type) 
    {

	case SOLVER_TYPE_MCSA: 
	{
	    // Set the PL
	    if( d_plist.get() ) 
	    {
		Teuchos::ParameterList &solverTypesPL = 
		    d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &mcsaPL = 
		    solverTypesPL.sublist(MCSA_name);
		solverPL = Teuchos::rcp( &mcsaPL, false );
	    }
	    // Create the solver
	    solver = 
		rcp(new MCLS::MCSASolverManager<Vector,Matrix>(
			global_comm, solverPL) );
	    iterativeSolver = Teuchos::rcp( 
		new MCLS::SolverManagerAdapter<MultiVector,Matrix>(solver) );
	    iterativeSolver->setProblem( lp );

	    break;
	}

	case SOLVER_TYPE_SEQUENTIAL_MC: 
	{
	    // Set the PL
	    if( d_plist.get() ) 
	    {
		Teuchos::ParameterList &solverTypesPL = 
		    d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &sequentialmcPL = 
		    solverTypesPL.sublist(SequentialMC_name);
		solverPL = Teuchos::rcp( &sequentialmcPL, false );
	    }
	    // Create the solver
	    solver = 
		rcp(new MCLS::SequentialMCSolverManager<Vector,Matrix>(
			global_comm, solverPL) );
	    iterativeSolver = Teuchos::rcp( 
		new MCLS::SolverManagerAdapter<MultiVector,Matrix>(solver) );
	    iterativeSolver->setProblem( lp );

	    break;
	}

	case SOLVER_TYPE_ADJOINT_MC: 
	{
	    // Set the PL
	    if( d_plist.get() ) 
	    {
		Teuchos::ParameterList &solverTypesPL = 
		    d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &adjointmcPL = 
		    solverTypesPL.sublist(AdjointMC_name);
		solverPL = Teuchos::rcp( &adjointmcPL, false );
	    }
	    // Create the solver
	    solver = 
		rcp(new MCLS::AdjointSolverManager<Vector,Matrix>(
			global_comm, solverPL) );
	    iterativeSolver = Teuchos::rcp( 
		new MCLS::SolverManagerAdapter<MultiVector,Matrix>(solver) );
	    iterativeSolver->setProblem( lp );

	    break;
	}

	case SOLVER_TYPE_FORWARD_MC: 
	{
	    // Set the PL
	    if( d_plist.get() ) 
	    {
		Teuchos::ParameterList &solverTypesPL = 
		    d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &forwardmcPL = 
		    solverTypesPL.sublist(ForwardMC_name);
		solverPL = Teuchos::rcp( &forwardmcPL, false );
	    }
	    // Create the solver
	    solver = 
		rcp(new MCLS::ForwardSolverManager<Vector,Matrix>(
			global_comm, solverPL) );
	    iterativeSolver = Teuchos::rcp( 
		new MCLS::SolverManagerAdapter<MultiVector,Matrix>(solver) );
	    iterativeSolver->setProblem( lp );

	    break;
	}

	case SOLVER_TYPE_FIXED_POINT: 
	{
	    // Set the PL
	    if( d_plist.get() ) 
	    {
		Teuchos::ParameterList &solverTypesPL = 
		    d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &fixedpointPL = 
		    solverTypesPL.sublist(FixedPoint_name);
		solverPL = Teuchos::rcp( &fixedpointPL, false );
	    }
	    // Create the solver
	    solver = 
		rcp(new MCLS::FixedPointSolverManager<Vector,Matrix>(
			global_comm, solverPL) );
	    iterativeSolver = Teuchos::rcp( 
		new MCLS::SolverManagerAdapter<MultiVector,Matrix>(solver) );
	    iterativeSolver->setProblem( lp );

	    break;
	}

	default:
	{
	    TEUCHOS_TEST_FOR_EXCEPT(true);
	}
    }

    // Initialize the LOWS object.
    Teuchos::RCP<MCLS::LinearProblemBase<Scalar> > lp_base = lp;
    lp_base->setOperator( fwdOp );
    if ( prec.get() )
    {
	lp_base->setLeftPrec( prec->getLeftPrecOp() );
	lp_base->setRightPrec( prec->getRightPrecOp() );
    }
    Teuchos::RCP<MCLS::SolverManagerBase<Scalar> > solver_base = iterativeSolver;
    mclsOp->initialize(	lp_base, solverPL, solver_base,
			fwdOpSrc, prec, my_prec.get()==NULL, approxFwdOpSrc,
			supportSolveUse	);
    mclsOp->setOStream(out);
    mclsOp->setVerbLevel(verbLevel);
#ifdef TEUCHOS_DEBUG
    if( d_plist.get( )) 
    {
	// Make sure we read the list correctly. Validate 0th and 1st level deep.
	d_plist->validateParameters(*this->getValidParameters(),1);
    }
#endif

    // Output.
    if( out.get() && 
	static_cast<int>(verbLevel) > static_cast<int>(Teuchos::VERB_LOW) )
    {
	*out << "\nLeaving Thyra::MCLSLinearOpWithSolveFactory<"
	     << ST::name() << ">::initializeOpImpl(...) ...\n";
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get an Epetra_RowMatrix from the linear operator source.
 */
template<class Scalar>
RCP<const Epetra_RowMatrix> 
MCLSLinearOpWithSolveFactory<Scalar>::getEpetraRowMatrix(
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    EpetraOperatorViewExtractorStd epetraFwdOpViewExtractor;
    RCP<const Epetra_Operator> epetraFwdOp;
    EOpTransp epetraFwdOpTransp;
    EApplyEpetraOpAs epetraFwdOpApplyAs;
    EAdjointEpetraOp epetraFwdOpAdjointSupport;
    double epetraFwdOpScalar;
    epetraFwdOpViewExtractor.getEpetraOpView(
	fwdOpSrc.getOp(), 
	Teuchos::outArg(epetraFwdOp), 
	Teuchos::outArg(epetraFwdOpTransp),
	Teuchos::outArg(epetraFwdOpApplyAs), 
	Teuchos::outArg(epetraFwdOpAdjointSupport),
	Teuchos::outArg(epetraFwdOpScalar)
	);

    return Teuchos::rcp_dynamic_cast<const Epetra_RowMatrix>(epetraFwdOp);
}

//---------------------------------------------------------------------------//
/*!
 * \brief Check for compatibility with Epetra.
 */
template<class Scalar>
bool MCLSLinearOpWithSolveFactory<Scalar>::isEpetraCompatible(
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    // MCLS interfaces are currently only implemented for Epetra_RowMatrix.
    RCP<const Epetra_RowMatrix> epetraFwdOp = 
	getEpetraRowMatrix( fwdOpSrc );

    bool row_matrix_compatible = Teuchos::nonnull( epetraFwdOp );

    return row_matrix_compatible;
}

//---------------------------------------------------------------------------//
/*!
 * \brief Get a Tpetra::CrsMatrix from the linear operator source.
 */
template<class Scalar>
template<class LO, class GO>
RCP<const Tpetra::CrsMatrix<Scalar,LO,GO> >
MCLSLinearOpWithSolveFactory<Scalar>::getTpetraCrsMatrix( 
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetraFwdOp =
	TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraOperator(
	    fwdOpSrc.getOp() );

    return Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<Scalar,LO,GO> >(
	tpetraFwdOp );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Check for compatibility with Tpetra.
 */
template<class Scalar>
template<class LO, class GO>
bool MCLSLinearOpWithSolveFactory<Scalar>::isTpetraCompatible( 
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    typedef Thyra::TpetraLinearOp<Scalar,LO,GO> TpetraOpType; 
    typedef Tpetra::CrsMatrix<Scalar,LO,GO> CrsType;

    RCP<const TpetraOpType> tpetra_op =	
	Teuchos::rcp_dynamic_cast<const TpetraOpType>( fwdOpSrc.getOp() );

    bool operator_compatible = Teuchos::nonnull(tpetra_op);

    if( operator_compatible )
    {
	RCP<const CrsType> crs = getTpetraCrsMatrix<LO,GO>( fwdOpSrc );
	return Teuchos::nonnull(crs);
    }

    return false;
}

//---------------------------------------------------------------------------//

} // namespace Thyra

//---------------------------------------------------------------------------//

#endif // THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_IMPL_HPP

//---------------------------------------------------------------------------//
// end ThyraMCLSLinearOpWithSolveFactory_impl.hpp
//---------------------------------------------------------------------------//
