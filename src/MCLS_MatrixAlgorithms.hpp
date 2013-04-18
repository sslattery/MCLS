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
 * \file MCLS_MatrixAlgorithms.hpp
 * \author Stuart R. Slattery
 * \brief Matrix algorithms definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MATRIXALGORITHMS_HPP
#define MCLS_MATRIXALGORITHMS_HPP

#include <Teuchos_RCP.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class UndefinedMatrixAlgorithms
 * \brief Class for undefined matrix algorithms. 
 *
 * Will throw a compile-time error if these algorithms are not specialized.
 */
template<class Vector, class Matrix>
struct UndefinedMatrixAlgorithms
{
    static inline void notDefined()
    {
	return Matrix::this_type_is_missing_a_specialization();
    }
};

//---------------------------------------------------------------------------//
/*!
 * \class MatrixAlgorithms
 * \brief Algorithms for matrices.
 *
 * MatrixAlgorithms defines an interface for parallel distributed
 * matrices. (e.g. Tpetra::CrsMatrix or Epetra_VbrMatrix).
 */
template<class Vector, class Matrix>
class MatrixAlgorithms
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                       vector_type;
    typedef Matrix                                       matrix_type;
    typedef typename Vector::scalar_type                 scalar_type;
    typedef typename Vector::local_ordinal_type          local_ordinal_type;
    typedef typename Vector::global_ordinal_type         global_ordinal_type;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new matrix filled with
     *  values from the given matrix to which the reduced domain approximation
     *  has been applied.
     */
    static void reducedDomainApproximation( 
        const Matrix& matrix,
        const double neumann_relax,
        const double filter_tol,
        const int fill_value,
        const double weight_recovery,
        Teuchos::RCP<Matrix>& reduced_H,
        Teuchos::RCP<Vector>& recovered_weights )
    { 
	UndefinedMatrixAlgorithms<Vector,Matrix>::notDefined(); 
	return Teuchos::null; 
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_MATRIXALGORITHMS_HPP

//---------------------------------------------------------------------------//
// end MCLS_MatrixAlgorithms.hpp
// ---------------------------------------------------------------------------//

