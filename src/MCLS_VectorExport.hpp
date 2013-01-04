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
 * \class UndefinedVectorExport
 * \brief Class for undefined vector export. 
 *
 * Will throw a compile-time error if the specified VectorExport functions are
 * not specialized.
 */
template<class VectorTraits>
struct UndefinedVectorExport
{
    static inline void notDefined()
    {
	return VectorTraits::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class VectorExport
 * \brief Parallel export mechanism.
 *
 * VectorExport defines an interface for moving data between parallel
 * distributed vectors with different parallel distributions.
 * (e.g. Tpetra::Vector or Epetra_Vector). We separate this from the stateless
 * traits classes as the parallel mappings are expensive to construct. This
 * provides a state container for these mappings so that the may be reused.
 *
 * The members should be overloaded for each vector type.
 */
template<class VectorTraits>
class VectorExport
{
  public:

    //@{
    //! Typedefs.
    typedef VectorTraits                                     VT;
    typedef typename VT::vector_type                         Vector;
    typedef typename VT::export_type                         Export;
    //@}

    /*!
     * \brief Constructor.
     */
    VectorExport( const Teuchos::RCP<Vector>& source_vector,
		  const Teuchos::RCP<Vector>& target_vector )
	: d_source_vector( source_vector )
	, d_target_vector( target_vector )
    { 
	setup();
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
	UndefinedVectorExport<VectorTraits>::notDefined();
    }

    /*!
     * \brief Do the export. Insert new values that do not exist.
     */
    void doExportInsert()
    {
	UndefinedVectorExport<VectorTraits>::notDefined();
    }

    /*!
     * \brief Do the export. Replace existing values with new values.
     */
    void doExportReplace()
    {
	UndefinedVectorExport<VectorTraits>::notDefined();
    }

    /*!
     * \brief Do the export. Replace existing values if its absolute value is
     * smaller than the absolute value of the new value.
     */
    void doExportAbsMax()
    {
	UndefinedVectorExport<VectorTraits>::notDefined();
    }

  private:

    // Setup the source-to-target exporter.
    void setup()
    {
	UndefinedVectorExport<VectorTraits>::notDefined();
    }

  private:

    // Source vector.
    Teuchos::RCP<Vector> d_source_vector;

    // Target vector.
    Teuchos::RCP<Vector> d_target_vector;

    // Source-to-target exporter.
    Export d_export;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_VECTOREXPORT_HPP

//---------------------------------------------------------------------------//
// end MCLS_VectorExport.hpp
//---------------------------------------------------------------------------//

