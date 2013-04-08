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

#ifndef THYRA_MCLSPRECONDITIONERFACTORY_HPP
#define THYRA_MCLSPRECONDITIONERFACTORY_HPP

#include "Thyra_PreconditionerFactoryBase.hpp"

#include <Epetra_RowMatrix.h>

#include <Tpetra_CrsMatrix.hpp>

namespace Thyra {

enum EMCLSPrecType {
    PREC_TYPE_POINT_JACOBI,
    PREC_TYPE_BLOCK_JACOBI,
    PREC_TYPE_ILUT,
    PREC_TYPE_PARASAILS
};

inline std::istream& operator>>(
    std::istream& is, EMCLSPrecType& sType)
{
    int intval;
    is >> intval;
    sType = (EMCLSPrecType)intval;
    return is;
}

/** \brief Concrete preconditioner factory subclass based on MCLS.
 *
 * ToDo: Finish documentation!
 */
template<class Scalar>
class MCLSPreconditionerFactory : public PreconditionerFactoryBase<Scalar> {
  public:

    /** \name Parameter names for Parameter List */
    //@{

    /** \brief . */
    static const std::string  PrecType_name;
    /** \brief . */           
    static const std::string  PrecType_default;
    /** \brief . */
    static const std::string  PrecTypes_name;
    /** \brief . */
    static const std::string  PointJacobi_name;
    /** \brief . */
    static const std::string  BlockJacobi_name;
    /** \brief . */
    static const std::string  ILUT_name;
    /** \brief . */
    static const std::string  ParaSails_name;

    /** @name Constructors/initializers/accessors */
    //@{

    /** \brief . */
    MCLSPreconditionerFactory();
    
    //@}

    /** @name Overridden from PreconditionerFactoryBase */
    //@{

    /** \brief . */
    bool isCompatible( const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;
    /** \brief . */
    bool applySupportsConj(EConj conj) const;
    /** \brief . */
    bool applyTransposeSupportsConj(EConj conj) const;
    /** \brief . */
    Teuchos::RCP<PreconditionerBase<Scalar> > createPrec() const;
    /** \brief . */
    void initializePrec(
	const Teuchos::RCP<const LinearOpSourceBase<Scalar> >    &fwdOpSrc
	,PreconditionerBase<Scalar>                              *prec
	,const ESupportSolveUse                                  supportSolveUse
	) const;
    /** \brief . */
    void uninitializePrec(
	PreconditionerBase<Scalar>                                *prec
	,Teuchos::RCP<const LinearOpSourceBase<Scalar> >          *fwdOpSrc
	,ESupportSolveUse                                         *supportSolveUse
	) const;

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
    template<class LO, class GO>
    bool isTpetraCompatible( const LinearOpSourceBase<Scalar> &fwdOpSrc ) const;

  private:

    // ////////////////////////////////
    // Private data members

    // Prec type.
    EMCLSPrecType d_prec_type;

    Teuchos::RCP<Teuchos::ParameterList>       d_plist;
};

} // namespace Thyra

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "Thyra_MCLSPreconditionerFactory_impl.hpp"

//---------------------------------------------------------------------------//

#endif // THYRA_MCLSPRECONDITIONERFACTORY_HPP

//---------------------------------------------------------------------------//
// end Thyra_MCLSPreconditionerFactory.hpp
//---------------------------------------------------------------------------//

