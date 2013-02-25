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
 * \file MCLS_EpetraVectorExport.hpp
 * \author Stuart R. Slattery
 * \brief Epetra::Vector Export.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_EPETRAVECTOREXPORT_HPP
#define MCLS_EPETRAVECTOREXPORT_HPP

#include <MCLS_VectorExport.hpp>
#include <MCLS_DBC.hpp>

#include <Teuchos_as.hpp>

#include <Epetra_Vector.h>
#include <Epetra_Export.h>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class VectorExport
 * \brief VectorExport specialization for Epetra::Vector.
 */
template<>
class VectorExport<Epetra_Vector>
{
  public:

    //@{
    //! Typedefs.
    typedef Epetra_Vector                           vector_type;
    typedef Epetra_Export                           export_type;
    //@}

    /*!
     * \brief Constructor.
     */
    VectorExport( const Teuchos::RCP<vector_type>& source_vector,
		  const Teuchos::RCP<vector_type>& target_vector )
	: d_source_vector( source_vector )
	, d_target_vector( target_vector )
	, d_export( new export_type( d_source_vector->Map(), 
				     d_target_vector->Map() ) )
    {
	MCLS_ENSURE( !d_source_vector.is_null() );
	MCLS_ENSURE( !d_target_vector.is_null() );
	MCLS_ENSURE( !d_export.is_null() );
    }

    /*!
     * \brief Destructor.
     */
    ~VectorExport()
    { /* ... */ }

    /*!
     * \brief Do the export. Existing values are summed with new values.
     */
    void doExportAdd()
    {
	d_target_vector->Export( *d_source_vector, *d_export, Add );
    }

    /*!
     * \brief Do the export. Insert new values that do not exist.
     */
    void doExportInsert()
    {
	d_target_vector->Export( *d_source_vector, *d_export, Insert );
    }

  private:

    // Source vector.
    Teuchos::RCP<vector_type> d_source_vector;

    // Target vector.
    Teuchos::RCP<vector_type> d_target_vector;

    // Source-to-target exporter.
    Teuchos::RCP<export_type> d_export;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_EPETRAVECTOREXPORT_HPP

//---------------------------------------------------------------------------//
// end MCLS_EpetraVectorExport.hpp
//---------------------------------------------------------------------------//
