/*@HEADER
// ***********************************************************************
// 
//       Ifpack: Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2002) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version.
//  
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ***********************************************************************
//@HEADER
*/

#include <MCLS_Preconditioner.hpp>
#include <MCLS_TpetraPointJacobiPreconditioner.hpp>
#include <MCLS_TpetraBlockJacobiPreconditioner.hpp>
#include <MCLS_EpetraPointJacobiPreconditioner.hpp>
#include <MCLS_EpetraBlockJacobiPreconditioner.hpp>
#include <MCLS_EpetraILUTPreconditioner.hpp>

#include "Teuchos_dyn_cast.hpp"
#include "Teuchos_implicit_cast.hpp"
#include "Teuchos_StandardParameterEntryValidators.hpp"
#include "Teuchos_VerboseObjectParameterListHelpers.hpp"
#include "Teuchos_ValidatorXMLConverterDB.hpp"
#include "Teuchos_StaticSetupMacro.hpp"
#include <Teuchos_as.hpp>

#include <Thyra_DefaultPreconditioner.hpp>
#include <Thyra_EpetraOperatorViewExtractorStd.hpp>
#include <Thyra_EpetraLinearOp.hpp>
#include <Thyra_TpetraThyraWrappers.hpp>

#include <Epetra_Operator.h>

#include <Tpetra_Operator.hpp>

namespace Thyra {

//---------------------------------------------------------------------------//
// Parameter names for Parameter List
template<class Scalar>
const std::string MCLSPreconditionerFactory<Scalar>::PrecType_name = 
    "Preconditioner Type";

template<class Scalar>
const std::string MCLSPreconditionerFactory<Scalar>::PrecType_default = 
    "Point Jacobi";

template<class Scalar>
const std::string MCLSPreconditionerFactory<Scalar>::PrecTypes_name = 
    "Preconditioner Types";

template<class Scalar>
const std::string MCLSPreconditionerFactory<Scalar>::PointJacobi_name = 
    "Point Jacobi";

template<class Scalar>
const std::string MCLSPreconditionerFactory<Scalar>::BlockJacobi_name = 
    "Block Jacobi";

template<class Scalar>
const std::string MCLSPreconditionerFactory<Scalar>::ILUT_name = 
    "ILUT";

// Constructors/initializers/accessors

//---------------------------------------------------------------------------//
template<class Scalar>
MCLSPreconditionerFactory<Scalar>::MCLSPreconditionerFactory()
    : d_prec_type(PREC_TYPE_POINT_JACOBI)
{
    getValidParameters(); // Make sure validators get created!
}

// Overridden from PreconditionerFactoryBase

//---------------------------------------------------------------------------//
template<class Scalar>
bool MCLSPreconditionerFactory<Scalar>::isCompatible(
    const LinearOpSourceBase<Scalar> &fwdOpSrc
    ) const
{
    // MCLS interfaces are currently only implemented for Epetra_RowMatrix and
    // Tpetra::CrsMatrix. 
    bool epetra_compatible = isEpetraCompatible( fwdOpSrc );
    bool tpetra_compatible = ( isTpetraCompatible<int,int>(fwdOpSrc) ||
			       isTpetraCompatible<int,long>(fwdOpSrc) );
    return ( epetra_compatible || tpetra_compatible );
}

//---------------------------------------------------------------------------//
template<class Scalar>
bool MCLSPreconditionerFactory<Scalar>::applySupportsConj(EConj conj) const
{
    return false;
}

//---------------------------------------------------------------------------//
template<class Scalar>
bool MCLSPreconditionerFactory<Scalar>::applyTransposeSupportsConj(EConj conj) const
{
    return false;
}

//---------------------------------------------------------------------------//
template<class Scalar>
Teuchos::RCP<PreconditionerBase<Scalar> >
MCLSPreconditionerFactory<Scalar>::createPrec() const
{
    return Teuchos::rcp(new DefaultPreconditioner<Scalar>());
}

//---------------------------------------------------------------------------//
template<class Scalar>
void MCLSPreconditionerFactory<Scalar>::initializePrec(
    const Teuchos::RCP<const LinearOpSourceBase<Scalar> >    &fwdOpSrc
    ,PreconditionerBase<Scalar>                                      *prec
    ,const ESupportSolveUse                                           supportSolveUse
    ) const
{
    using Teuchos::outArg;
    using Teuchos::OSTab;
    using Teuchos::dyn_cast;
    using Teuchos::RCP;
    using Teuchos::null;
    using Teuchos::rcp;
    using Teuchos::rcp_dynamic_cast;
    using Teuchos::rcp_const_cast;
    using Teuchos::set_extra_data;
    using Teuchos::get_optional_extra_data;
    using Teuchos::implicit_cast;

    const Teuchos::RCP<Teuchos::FancyOStream> out       = this->getOStream();
    const Teuchos::EVerbosityLevel                    verbLevel = this->getVerbLevel();
    Teuchos::OSTab tab(out);
    if(out.get() && implicit_cast<int>(verbLevel) > implicit_cast<int>(Teuchos::VERB_LOW))
	*out << "\nEntering Thyra::MCLSPreconditionerFactory<Scalar>::initializePrec(...) ...\n";
#ifdef TEUCHOS_DEBUG
    TEUCHOS_TEST_FOR_EXCEPT(fwdOpSrc.get()==NULL);
    TEUCHOS_TEST_FOR_EXCEPT(prec==NULL);
#endif
    Teuchos::RCP<const LinearOpBase<Scalar> >
	fwdOp = fwdOpSrc->getOp();
#ifdef TEUCHOS_DEBUG
    TEUCHOS_TEST_FOR_EXCEPT(fwdOp.get()==NULL);
#endif

    // Build the preconditioner.
    DefaultPreconditioner<Scalar>
	*defaultPrec = &Teuchos::dyn_cast<DefaultPreconditioner<Scalar> >(*prec);

    if ( isEpetraCompatible(*fwdOpSrc) )
    {
	Teuchos::RCP<MCLS::Preconditioner<Epetra_RowMatrix> > mcls_prec;

	// Point Jacobi.
	if ( d_prec_type == PREC_TYPE_POINT_JACOBI )
	{
            // Build.
	    mcls_prec = Teuchos::rcp( new MCLS::EpetraPointJacobiPreconditioner() );
            mcls_prec->setOperator( getEpetraRowMatrix(*fwdOpSrc) );
            mcls_prec->buildPreconditioner();

            // Left.
            Teuchos::RCP<EpetraLinearOp> epetra_op = 
                Teuchos::rcp( new EpetraLinearOp() );
            epetra_op->initialize( 
                Teuchos::rcp_const_cast<Epetra_RowMatrix>(
                    mcls_prec->getLeftPreconditioner()) );
            Teuchos::RCP<const LinearOpBase<Scalar> > thyra_op = epetra_op;
            defaultPrec->initializeLeft( thyra_op );

	}
	
	// Block Jacobi.
	else if ( d_prec_type == PREC_TYPE_BLOCK_JACOBI )
	{
            // Setup.
	    Teuchos::ParameterList &precTypesPL = 
		d_plist->sublist(PrecTypes_name);
	    Teuchos::ParameterList &blockJacobiPL = 
		precTypesPL.sublist(BlockJacobi_name);
	    Teuchos::RCP<Teuchos::ParameterList> prec_plist = 
		Teuchos::rcp( &blockJacobiPL, false );

            // Build.
	    mcls_prec = Teuchos::rcp( 
		new MCLS::EpetraBlockJacobiPreconditioner(prec_plist) );
            mcls_prec->setOperator( getEpetraRowMatrix(*fwdOpSrc) );
            mcls_prec->buildPreconditioner();

            // Left.
            Teuchos::RCP<EpetraLinearOp> epetra_op = 
                Teuchos::rcp( new EpetraLinearOp() );
            epetra_op->initialize( 
                Teuchos::rcp_const_cast<Epetra_RowMatrix>(
                    mcls_prec->getLeftPreconditioner()) );
            Teuchos::RCP<const LinearOpBase<Scalar> > thyra_op = epetra_op;
            defaultPrec->initializeLeft( thyra_op );
	}

        // ILUT.
	else if ( d_prec_type == PREC_TYPE_ILUT )
	{
            // Setup.
	    Teuchos::ParameterList &precTypesPL = 
		d_plist->sublist(PrecTypes_name);
	    Teuchos::ParameterList &ilutPL = 
		precTypesPL.sublist(ILUT_name);
	    Teuchos::RCP<Teuchos::ParameterList> prec_plist = 
		Teuchos::rcp( &ilutPL, false );
            
            // Build.
	    mcls_prec = Teuchos::rcp( 
		new MCLS::EpetraILUTPreconditioner(prec_plist) );
            mcls_prec->setOperator( getEpetraRowMatrix(*fwdOpSrc) );
            mcls_prec->buildPreconditioner();

            // Left
            Teuchos::RCP<EpetraLinearOp> epetra_lop = 
                Teuchos::rcp( new EpetraLinearOp() );
            epetra_lop->initialize( 
                Teuchos::rcp_const_cast<Epetra_RowMatrix>(
                    mcls_prec->getLeftPreconditioner()) );
            Teuchos::RCP<const LinearOpBase<Scalar> > thyra_lop = epetra_lop;

            // Right
            Teuchos::RCP<EpetraLinearOp> epetra_rop = 
                Teuchos::rcp( new EpetraLinearOp() );
            epetra_rop->initialize( 
                Teuchos::rcp_const_cast<Epetra_RowMatrix>(
                    mcls_prec->getRightPreconditioner()) );
            Teuchos::RCP<const LinearOpBase<Scalar> > thyra_rop = epetra_rop;

            // Initialize.
            defaultPrec->initializeLeftRight( thyra_lop, thyra_rop );
	}

    }
    else if ( isTpetraCompatible<int,int>(*fwdOpSrc) )
    {
	typedef int LO;
	typedef int GO;
	typedef Tpetra::CrsMatrix<Scalar,LO,GO> Matrix;

	Teuchos::RCP<MCLS::Preconditioner<Matrix> > mcls_prec;

	// Point Jacobi.
	if ( d_prec_type == PREC_TYPE_POINT_JACOBI )
	{
	    mcls_prec = Teuchos::rcp( 
		new MCLS::TpetraPointJacobiPreconditioner<Scalar,LO,GO>() );
	}
	
	// Block Jacobi.
	else if ( d_prec_type == PREC_TYPE_BLOCK_JACOBI )
	{
	    Teuchos::ParameterList &precTypesPL = 
		d_plist->sublist(PrecTypes_name);
	    Teuchos::ParameterList &blockJacobiPL = 
		precTypesPL.sublist(BlockJacobi_name);
	    Teuchos::RCP<Teuchos::ParameterList> prec_plist = 
		Teuchos::rcp( &blockJacobiPL, false );
	    mcls_prec = Teuchos::rcp( 
		new MCLS::TpetraBlockJacobiPreconditioner<Scalar,LO,GO>(prec_plist) );
	}

	// For now, we just have left preconditioners implemented so we do
	// this here.
	mcls_prec->setOperator( getTpetraCrsMatrix<LO,GO>(*fwdOpSrc) );
	mcls_prec->buildPreconditioner();
	Teuchos::RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetra_op = 
	    mcls_prec->getLeftPreconditioner();
	Teuchos::RCP<const LinearOpBase<Scalar> > thyra_op = 
	    Thyra::createConstLinearOp<Scalar,LO,GO>( tpetra_op );
	defaultPrec->initializeLeft( thyra_op );
    }
    else if ( isTpetraCompatible<int,long>(*fwdOpSrc) )
    {
	typedef int LO;
	typedef long GO;
	typedef Tpetra::CrsMatrix<Scalar,LO,GO> Matrix;

	Teuchos::RCP<MCLS::Preconditioner<Matrix> > mcls_prec;

	// Point Jacobi.
	if ( d_prec_type == PREC_TYPE_POINT_JACOBI )
	{
	    mcls_prec = Teuchos::rcp( 
		new MCLS::TpetraPointJacobiPreconditioner<Scalar,LO,GO>() );
	}
	
	// Block Jacobi.
	else if ( d_prec_type == PREC_TYPE_BLOCK_JACOBI )
	{
	    Teuchos::ParameterList &precTypesPL = 
		d_plist->sublist(PrecTypes_name);
	    Teuchos::ParameterList &blockJacobiPL = 
		precTypesPL.sublist(BlockJacobi_name);
	    Teuchos::RCP<Teuchos::ParameterList> prec_plist = 
		Teuchos::rcp( &blockJacobiPL, false );
	    mcls_prec = Teuchos::rcp( 
		new MCLS::TpetraBlockJacobiPreconditioner<Scalar,LO,GO>(prec_plist) );
	}

	// For now, we just have left preconditioners implemented so we do
	// this here.
	mcls_prec->setOperator( getTpetraCrsMatrix<LO,GO>(*fwdOpSrc) );
	mcls_prec->buildPreconditioner();
	Teuchos::RCP<const Tpetra::Operator<Scalar,LO,GO> > tpetra_op = 
	    mcls_prec->getLeftPreconditioner();
	Teuchos::RCP<const LinearOpBase<Scalar> > thyra_op = 
	    Thyra::createConstLinearOp<Scalar,LO,GO>( tpetra_op );
	defaultPrec->initializeLeft( thyra_op );
    }
    else
    {
	TEUCHOS_TEST_FOR_EXCEPT(true);
    }
}

//---------------------------------------------------------------------------//
template<class Scalar>
void MCLSPreconditionerFactory<Scalar>::uninitializePrec(
    PreconditionerBase<Scalar>                                *prec
    ,Teuchos::RCP<const LinearOpSourceBase<Scalar> >  *fwdOpSrc
    ,ESupportSolveUse                                         *supportSolveUse
    ) const
{
    TEUCHOS_TEST_FOR_EXCEPT(true); // ToDo: Implement when needed!
}

// Overridden from ParameterListAcceptor

//---------------------------------------------------------------------------//
template<class Scalar>
void MCLSPreconditionerFactory<Scalar>::setParameterList(
    Teuchos::RCP<Teuchos::ParameterList> const& paramList)
{
    TEUCHOS_TEST_FOR_EXCEPT( Teuchos::is_null(paramList) );

    paramList->validateParametersAndSetDefaults(*this->getValidParameters(), 1);
    d_plist = paramList;
    d_prec_type =
	Teuchos::getIntegralValue<EMCLSPrecType>(*d_plist, PrecType_name);
    Teuchos::readVerboseObjectSublist(&*d_plist,this);
}

//---------------------------------------------------------------------------//
template<class Scalar>
Teuchos::RCP<Teuchos::ParameterList>
MCLSPreconditionerFactory<Scalar>::getNonconstParameterList()
{
    return d_plist;
}

//---------------------------------------------------------------------------//
template<class Scalar>
Teuchos::RCP<Teuchos::ParameterList>
MCLSPreconditionerFactory<Scalar>::unsetParameterList()
{
    Teuchos::RCP<Teuchos::ParameterList> d_plist = d_plist;
    d_plist = Teuchos::null;
    return d_plist;
}

//---------------------------------------------------------------------------//
template<class Scalar>
Teuchos::RCP<const Teuchos::ParameterList>
MCLSPreconditionerFactory<Scalar>::getParameterList() const
{
    return d_plist;
}

//---------------------------------------------------------------------------//
template<class Scalar>
Teuchos::RCP<const Teuchos::ParameterList>
MCLSPreconditionerFactory<Scalar>::getValidParameters() const
{
    using Teuchos::as;
    using Teuchos::tuple;
    using Teuchos::setStringToIntegralParameter;
    Teuchos::ValidatorXMLConverterDB::addConverter(
	Teuchos::DummyObjectGetter<
	Teuchos::StringToIntegralParameterEntryValidator<EMCLSPrecType> 
	>::getDummyObject(),
	Teuchos::DummyObjectGetter<Teuchos::StringToIntegralValidatorXMLConverter<
	EMCLSPrecType> >::getDummyObject());

    static RCP<Teuchos::ParameterList> validParamList;
    if( validParamList.get()==NULL ) 
    {
	validParamList = Teuchos::rcp(new Teuchos::ParameterList(
					  "MCLSPreconditionerFactory") );

	setStringToIntegralParameter<EMCLSPrecType>(
	    PrecType_name, PrecType_default,
	    "Type of preconditioning algorithm to use.",
	    tuple<std::string>(
		"Point Jacobi",
		"Block Jacobi",
                "ILUT"
		),
	    tuple<std::string>(
		"Point Jacobi preconditioning - Left scales the linear operator"
		"by the inverse of its diagonal",

		"Block Jacobi preconditioning - Left scales the linear operator"
		"by the inverse of its diagonal blocks. Blocks must be local"
		"and are of a user-specified size",

                "Incomplete LU factorization with threshold - Left/Right"
                "preconditioning for the linear operator"
		),
	    tuple<EMCLSPrecType>(
		PREC_TYPE_POINT_JACOBI,
		PREC_TYPE_BLOCK_JACOBI,
                PREC_TYPE_ILUT
		),
	    &*validParamList
	    );

	// We'll use Epetra to get the valid parameters for the preconditioner
	// subclasses as they are independent of the vector/operator
	// implementation used.
	Teuchos::ParameterList
	    &precTypesSL = validParamList->sublist(PrecTypes_name);
	{
	    MCLS::EpetraPointJacobiPreconditioner prec;
	    precTypesSL.sublist(PointJacobi_name).setParameters(
		*(prec.getValidParameters()) );
	}
	{
	    MCLS::EpetraBlockJacobiPreconditioner prec(Teuchos::parameterList());
	    precTypesSL.sublist(BlockJacobi_name).setParameters(
		*(prec.getValidParameters()) );
	}
	{
	    MCLS::EpetraILUTPreconditioner prec(Teuchos::parameterList());
	    precTypesSL.sublist(ILUT_name).setParameters(
		*(prec.getValidParameters()) );
	}
    }

    return validParamList;
}

// Public functions overridden from Teuchos::Describable

//---------------------------------------------------------------------------//
template<class Scalar>
std::string MCLSPreconditionerFactory<Scalar>::description() const
{
    std::ostringstream oss;
    oss << "Thyra::MCLSPreconditionerFactory";
    return oss.str();
}

// private
//---------------------------------------------------------------------------//
/*!
 * \brief Get an Epetra_RowMatrix from the linear operator source.
 */
template<class Scalar>
RCP<const Epetra_RowMatrix> 
MCLSPreconditionerFactory<Scalar>::getEpetraRowMatrix(
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
bool MCLSPreconditionerFactory<Scalar>::isEpetraCompatible(
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
MCLSPreconditionerFactory<Scalar>::getTpetraCrsMatrix( 
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
bool MCLSPreconditionerFactory<Scalar>::isTpetraCompatible( 
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
