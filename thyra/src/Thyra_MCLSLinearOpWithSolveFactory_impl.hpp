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

#include <Thyra_MCLSLinearOpWithSolve.hpp>

#include <MCLS_DBC.hpp>
#include <MCLS_MCSASolverManager.hpp>
#include <MCLS_SequentialMCSolverManager.hpp>
#include <MCLS_AdjointMCSolverManager.hpp>

#include "MCLSLinearProblemAdapter.hpp"
#include "MCLSSolverManagerAdapter.hpp"

#include <Teuchos_VerboseObjectParameterListHelpers.hpp>
#include <Teuchos_StandardParameterEntryValidators.hpp>
#include <Teuchos_dyn_cast.hpp>
#include <Teuchos_ValidatorXMLConverterDB.hpp>
#include <Teuchos_StandardValidatorXMLConverters.hpp>
#include <Teuchos_as.hpp>

#include <Thyra_EpetraOperatorViewExtractorStd.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

#include <Epetra_Operator.h>

#include <Tpetra_Operator.hpp>

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
     // For now, MCLS doesn't handle preconditioners.
     return false;
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
    Require( Teuchos::nonnull( precFactory );

    RCP<const Teuchos::ParameterList> precFactory_valid_plist = 
	     precFactory->getValidParameters();
    const std::string d_precFactoryName =
	( precFactoryName != ""
	  ? precFactoryName
	  : ( precFactory_valid_plist.get() ? 
	      precFactory_valid_plist->name() : 
	      "GENERIC PRECONDITIONER FACTORY" ) 
	    )
	);
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
    if(precFactory) *precFactory = d_prec_Factory;
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
    // Check the preconditioner factory for compatibility. We don't use them
    // with MCLS yet but we should check for consistency.
    bool prec_compatible = true;
    if( Teuchos::nonnull(prec_factory) )
    {
	prec_compatible = d_prec_factory->isCompatible(fwdOpSrc);
    }

    // MCLS interfaces are currently only implemented for Epetra_RowMatrix and
    // Tpetra::CrsMatrix. 
    bool epetra_compatible = isEpetraCompatible( fwdOpSrc );
    bool tpetra_compatible = isTpetraCompatible( fwdOpSrc );  
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
    initializeOpImpl(fwdOpSrc,null,null,false,Op,supportSolveUse);
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
    initializeOpImpl(fwdOpSrc,null,null,true,Op,SUPPORT_SOLVE_UNSPECIFIED);
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
    initializeOpImpl(fwdOpSrc,null,prec,false,Op,supportSolveUse);
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
    initializeOpImpl(fwdOpSrc,approxFwdOpSrc,null,false,Op,supportSolveUse);
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
    Require( Op != NULL );

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
    Require( Teuchos::nonnull(paramList) );

    paramList->validateParametersAndSetDefaults(*this->getValidParameters(), 1);
    d_plist = paramList;
    d_solver_type =
	Teuchos::getIntegralValue<EMCLSSolverType>(*d_plist, SolverType_name);
    convergenceTestFrequency_ =
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
		"Adjoint MC"
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
		"right-hand sides sequentially."
		),
	    tuple<EMCLSSolverType>(
		SOLVER_TYPE_MCSA,
		SOLVER_TYPE_SEQUENTIAL_MC,
		SOLVER_TYPE_ADJOINT_MC
		),
	    &*validParamList
	    );
	validParamList->set(
	    ConvergenceTestFrequency_name, as<int>(1),
	    "Number of linear solver iterations to skip between applying"
	    " user-defined convergence test.");

	// We'll use Epetra to get the valid parameters for the solvers as
	// they are independent of the vector/operator implementation used.
	Teuchos::ParameterList
	    &solverTypesSL = validParamList->sublist(SolverTypes_name);
	{
	    MCLS::MCSASolverManager<Epetra_Vector,Epetra_RowMatrix> mgr;
	    solverTypesSL.sublist(MCSA_name).setParameters(
		*mgr.getValidParameters() );
	}
	{
	    MCLS::SequentialMCSolverManager<Epetra_Vector,Epetra_RowMatrix> mgr;
	    solverTypesSL.sublist(SequentialMC_name).setParameters(
		*mgr.getValidParameters() );
	}
	{
	    MCLS::AdjointMCSolverManager<Epetra_Vector,Epetra_RowMatrix> mgr;
	    solverTypesSL.sublist(AdjointMC_name).setParameters(
		*mgr.getValidParameters() );
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
 * \brief Initialize the linear solver.
 */
template<class Scalar>
void MCLSLinearOpWithSolveFactory<Scalar>::initializeOpImpl(
    const RCP<const LinearOpSourceBase<Scalar> >& fwdOpSrc,
    const RCP<const LinearOpSourceBase<Scalar> >& approxFwdOpSrc,
    const RCP<const PreconditionerBase<Scalar> >& prec_in,
    const bool reusePrec,
    LinearOpWithSolveBase<Scalar>* Op,
    const ESupportSolveUse supportSolveUse ) const
{
    using Teuchos::rcp;
    using Teuchos::set_extra_data;
    typedef Teuchos::ScalarTraits<Scalar> ST;
    typedef typename ST::magnitudeType ScalarMag;
    typedef MultiVectorBase<Scalar> MV_t;
    typedef LinearOpBase<Scalar> LO_t;

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

    // Get/Create the preconditioner
    RCP<PreconditionerBase<Scalar> > myPrec = Teuchos::null;
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
	    myPrec = ( 
		!mclsOp->isExternalPrec()
		? Teuchos::rcp_const_cast<PreconditionerBase<Scalar> >(
		    mclsOp->extract_prec()) : Teuchos::null );
	    bool hasExistingPrec = false;
	    if( myPrec.get() ) 
	    {
		hasExistingPrec = true;
		// ToDo: Get the forward operator and validate that it is the same
		// operator that is used here!
	    }
	    else 
	    {
		hasExistingPrec = false;
		myPrec = d_prec_factory->createPrec();
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
		    d_prec_factory->initializePrec(approxFwdOpSrc,&*myPrec);
		}
		else
		{
		    d_prec_factory->initializePrec(fwdOpSrc,&*myPrec);
		}
	    }
	    prec = myPrec;
	}
    }

    // Uninitialize the current solver object
    bool oldIsExternalPrec = false;
    RCP<MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> > oldLP = Teuchos::null;
    RCP<MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> > oldIterSolver = Teuchos::null;
    RCP<const LinearOpSourceBase<Scalar> > oldFwdOpSrc = Teuchos::null;
    RCP<const LinearOpSourceBase<Scalar> > oldApproxFwdOpSrc = Teuchos::null;   
    ESupportSolveUse oldSupportSolveUse = SUPPORT_SOLVE_UNSPECIFIED;
    mclsOp->uninitialize( &oldLP, NULL, &oldIterSolver, &oldFwdOpSrc,
			  NULL, &oldIsExternalPrec, &oldApproxFwdOpSrc, 
			  &oldSupportSolveUse );

    //
    // Create the MCLS linear problem
    // NOTE:  If one exists already, reuse it.
    //

    typedef MCLS::LinearProblemAdapter<Scalar,MV_t,LO_t> LP_t;
    RCP<LP_t> lp;
    if (oldLP != Teuchos::null) {
	lp = oldLP;
    }
    else {
	lp = rcp(new LP_t());
    }

    //
    // Set the operator
    //

    lp->setOperator(fwdOp);

    //
    // Set the preconditioner
    //

    if(prec.get()) {
	RCP<const LinearOpBase<Scalar> > unspecified = prec->getUnspecifiedPrecOp();
	RCP<const LinearOpBase<Scalar> > left = prec->getLeftPrecOp();
	RCP<const LinearOpBase<Scalar> > right = prec->getRightPrecOp();
	TEUCHOS_TEST_FOR_EXCEPTION(
	    !( left.get() || right.get() || unspecified.get() ), std::logic_error
	    ,"Error, at least one preconditoner linear operator objects must be set!"
	    );
	if(unspecified.get()) {
	    lp->setRightPrec(unspecified);
	    // ToDo: Allow user to determine whether this should be placed on the
	    // left or on the right through a parameter in the parameter list!
	}
	else {
	    // Set a left, right or split preconditioner
	    TEUCHOS_TEST_FOR_EXCEPTION(
		left.get(),std::logic_error
		,"Error, we can not currently handle a left preconditioner!"
		);
	    lp->setRightPrec(right);
	}
    }
    if(myPrec.get()) {
	set_extra_data<RCP<PreconditionerBase<Scalar> > >(myPrec,"MCLS::InternalPrec",
							  Teuchos::inOutArg(lp), Teuchos::POST_DESTROY, false);
    }
    else if(prec.get()) {
	set_extra_data<RCP<const PreconditionerBase<Scalar> > >(prec,"MCLS::ExternalPrec",
								Teuchos::inOutArg(lp), Teuchos::POST_DESTROY, false);
    }

    //
    // Generate the parameter list.
    //

    typedef MCLS::SolverManagerAdapter<Scalar,MV_t,LO_t> IterativeSolver_t;
    RCP<IterativeSolver_t> iterativeSolver = Teuchos::null;
    RCP<Teuchos::ParameterList> solverPL = Teuchos::rcp( new Teuchos::ParameterList() );
  
    switch(d_solver_type) {
	case SOLVER_TYPE_BLOCK_GMRES: 
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &gmresPL = solverTypesPL.sublist(BlockGMRES_name);
		solverPL = Teuchos::rcp( &gmresPL, false );
	    }
	    // Create the solver
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    } 
	    else {
		iterativeSolver = rcp(new MCLS::BlockGmresSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}
	case SOLVER_TYPE_PSEUDO_BLOCK_GMRES:
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &pbgmresPL = solverTypesPL.sublist(PseudoBlockGMRES_name);
		solverPL = Teuchos::rcp( &pbgmresPL, false );
	    }
	    // 
	    // Create the solver
	    // 
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    }
	    else {
		iterativeSolver = rcp(new MCLS::PseudoBlockGmresSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}
	case SOLVER_TYPE_BLOCK_CG:
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &cgPL = solverTypesPL.sublist(BlockCG_name);
		solverPL = Teuchos::rcp( &cgPL, false );
	    }
	    // Create the solver
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    }
	    else {
		iterativeSolver = rcp(new MCLS::BlockCGSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}
	case SOLVER_TYPE_PSEUDO_BLOCK_CG:
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &pbcgPL = solverTypesPL.sublist(PseudoBlockCG_name);
		solverPL = Teuchos::rcp( &pbcgPL, false );
	    }
	    // 
	    // Create the solver
	    // 
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    }
	    else {
		iterativeSolver = rcp(new MCLS::PseudoBlockCGSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}
	case SOLVER_TYPE_GCRODR:
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &gcrodrPL = solverTypesPL.sublist(GCRODR_name);
		solverPL = Teuchos::rcp( &gcrodrPL, false );
	    }
	    // Create the solver
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    } 
	    else {
		iterativeSolver = rcp(new MCLS::GCRODRSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}
	case SOLVER_TYPE_RCG:
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &rcgPL = solverTypesPL.sublist(RCG_name);
		solverPL = Teuchos::rcp( &rcgPL, false );
	    }
	    // Create the solver
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    } 
	    else {
		iterativeSolver = rcp(new MCLS::RCGSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}
	case SOLVER_TYPE_MINRES:
	{
	    // Set the PL
	    if(d_plist.get()) {
		Teuchos::ParameterList &solverTypesPL = d_plist->sublist(SolverTypes_name);
		Teuchos::ParameterList &minresPL = solverTypesPL.sublist(MINRES_name);
		solverPL = Teuchos::rcp( &minresPL, false );
	    }
	    // Create the solver
	    if (oldIterSolver != Teuchos::null) {
		iterativeSolver = oldIterSolver;
		iterativeSolver->setProblem( lp );
		iterativeSolver->setParameters( solverPL );
	    }
	    else {
		iterativeSolver = rcp(new MCLS::MinresSolMgr<Scalar,MV_t,LO_t>(lp,solverPL));
	    }
	    break;
	}

	default:
	{
	    TEUCHOS_TEST_FOR_EXCEPT(true);
	}
    }

    //
    // Initialize the LOWS object
    //

    mclsOp->initialize(
	lp, solverPL, iterativeSolver,
	fwdOpSrc, prec, myPrec.get()==NULL, approxFwdOpSrc,
	supportSolveUse, convergenceTestFrequency_
	);
    mclsOp->setOStream(out);
    mclsOp->setVerbLevel(verbLevel);
#ifdef TEUCHOS_DEBUG
    if(d_plist.get()) {
	// Make sure we read the list correctly
	d_plist->validateParameters(*this->getValidParameters(),1); // Validate 0th and 1st level deep
    }
#endif
    if(out.get() && static_cast<int>(verbLevel) > static_cast<int>(Teuchos::VERB_LOW))
	*out << "\nLeaving Thyra::MCLSLinearOpWithSolveFactory<"<<ST::name()<<">::initializeOpImpl(...) ...\n";
  
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
template<class Scalar, class LO, class GO>
RCP<Tpetra::CrsMatrix<Scalar,LO,GO> >
MCLSLinearOpWithSolveFactory<Scalar>::getTpetraCrsMatrix( 
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetraFwdOp =
	TpetraOperatorVectorExtraction<Scalar,LO,GO>::getConstTpetraOperator(
	    fwdOpSrc->getOp() );

    return Teuchos::rcp_dynamic_cast<const Tpetra::CrsMatrix<Scalar,LO,GO> >(
	tpetraFwdOp );
}

//---------------------------------------------------------------------------//
/*!
 * \brief Check for compatibility with Tpetra.
 */
template<class Scalar>
bool MCLSLinearOpWithSolveFactory<Scalar>::isTpetraCompatible( 
    const LinearOpSourceBase<Scalar> &fwdOpSrc ) const
{
    // MCLS interfaces are currently only implemented for
    // Tpetra::CrsMatrix. For now, we'll only check compatibly with those
    // types enabled by Tpetra ETI. This is a pretty bad hack; we'll have to
    // rethink how we want to get the concrete types out of Tpetra to drive
    // MCLS. 
    RCP<Tpetra::CrsMatrix<int,int,double> > crs_i_i_d = 
	getTpetraCrsMatrix<int,int,d>( fwdOpSrc );
    RCP<Tpetra::CrsMatrix<int,long,double> > crs_i_l_d = 
	getTpetraCrsMatrix<int,long,d>( fwdOpSrc );

    bool tpetra_compatible = ( Teuchos::nonnull(crs_i_i_d) ||
			       Teuchos::nonnull(crs_i_l_d) );

    return tpetra_compatible;
}

//---------------------------------------------------------------------------//

} // namespace Thyra

//---------------------------------------------------------------------------//

#endif // THYRA_MCLS_LINEAR_OP_WITH_SOLVE_FACTORY_HPP

//---------------------------------------------------------------------------//
// end ThyraMCLSLinearOpWithSolveFactory.hpp
//---------------------------------------------------------------------------//
