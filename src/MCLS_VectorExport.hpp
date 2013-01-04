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
 * \file MCLS_VectorExport.hpp
 * \author Stuart R. Slattery
 * \brief Vector Export base class.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_VECTOREXPORT_HPP
#define MCLS_VECTOREXPORT_HPP

#include <Teuchos_RCP.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class VectorExport
 * \brief Export mechanism.
 *
 * VectorExport defines an interface for moving data between parallel
 * distributed vectors with different parallel distributions.
 * (e.g. Tpetra::Vector or Epetra_Vector).
 */
template<class Vector>
class VectorExport
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                  vector_type;
    //@}

    /*!
     * \brief Combine mode enum.
     */
    enum CombineMode {
	ADD,     /*!< Existing values will be summed with new values. */
	INSERT,  /*!< Insert new values that don't currently exist. */
	REPLACE, /*!< Existing values will be replaced with new values. */
	ABSMAX   /*!< Replacment is <tt>max( abs(old_value), abs(new_value) )</tt> */
    };

    /*!
     * \brief Constructor.
     */
    VectorExport( const Teuchos::RCP<Vector>& source_vector,
		  const Teuchos::RCP<Vector>& target_vector );

    /*!
     * \brief Destructor.
     */
    virtual ~VectorExport();

    /*!
     * \brief Do the export.
     */
    virtual doExport( CombineMode mode ) = 0;

  protected:

    // Source vector.
    Teuchos::RCP<Vector> b_source_vector;

    // Target vector.
    Teuchos::RCP<Vector> b_target_vector;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_VECTOREXPORT_HPP

//---------------------------------------------------------------------------//
// end MCLS_VectorExport.hpp
//---------------------------------------------------------------------------//

