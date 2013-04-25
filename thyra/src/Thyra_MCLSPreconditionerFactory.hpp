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
 * \file Thyra_MCLSPreconditionerFactory_impl.hpp
 * \author Stuart R. Slattery
 * \brief Thyra Preconditioner factory for MCLS.
 */
//---------------------------------------------------------------------------//

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
    PREC_TYPE_PARASAILS,
    PREC_TYPE_PSILUT,
    PREC_TYPE_ML
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
    /** \brief . */
    static const std::string  PSILUT_name;
    /** \brief . */
    static const std::string  ML_name;

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

