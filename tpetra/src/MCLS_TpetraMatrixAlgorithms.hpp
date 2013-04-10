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
 * \file MCLS_TpetraMatrixAlgorithms.hpp
 * \author Stuart R. Slattery
 * \brief Tpetra Matrix algorithms.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_TPETRAMATRIXALGORITHMS_HPP
#define MCLS_TPETRAMATRIXALGORITHMS_HPP

#include <algorithm>

#include <MCLS_DBC.hpp>
#include <MCLS_MatrixAlgorithms.hpp>
#include <MCLS_TpetraHelpers.hpp>
#include <MCLS_VectorTraits.hpp>
#include <MCLS_MatrixTraits.hpp>

#include <Teuchos_as.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayRCP.hpp>

#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Map.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MatrixAlgorithms
 * \brief Algorithms specialization for Tpetra_RowMatrix.
 */
template<class Scalar, class LO, class GO>
class MatrixAlgorithms<Tpetra::Vector<Scalar,LO,GO>, 
                       Tpetra::CrsMatrix<Scalar,LO,GO> >
{
  public:

    //@{
    //! Typedefs.
    typedef typename Tpetra::CrsMatrix<Scalar,LO,GO>      matrix_type;
    typedef typename Tpetra::Vector<Scalar,LO,GO>         vector_type;
    typedef typename vector_type::scalar_type             scalar_type;
    typedef typename vector_type::local_ordinal_type      local_ordinal_type;
    typedef typename vector_type::global_ordinal_type     global_ordinal_type;
    typedef VectorTraits<vector_type>                     VT;
    typedef MatrixTraits<vector_type,matrix_type>         MT;
    //@}

    /*!
     * \brief Create a reference-counted pointer to a new matrix filled with
     *  values from the given matrix to which the reduced domain approximation
     *  has been applied.
     */
    static void reducedDomainApproximation( 
        const matrix_type& matrix,
        const double filter_tol,
        const int fill_value,
        const double weight_recovery,
        Teuchos::RCP<matrix_type>& reduced_H,
        Teuchos::RCP<vector_type>& recovered_weights )
    { 
        reduced_H = Tpetra::createCrsMatrix<Scalar,LO,GO>( matrix.getRowMap() );
        recovered_weights = MT::cloneVectorFromMatrixRows( *reduced_H );

        Teuchos::ArrayRCP<double> rweights_view = 
            VT::viewNonConst( *recovered_weights );
        LO local_num_rows = MT::getLocalNumRows( matrix );
        GO global_row = 0;
        GO max_entries = MT::getGlobalMaxNumRowEntries( matrix );
        std::size_t num_entries = 0;
        Scalar filter_sum = 0.0;
        Teuchos::Array<Scalar> sorted_values;
        typename Teuchos::Array<Scalar>::iterator sorted_values_it;
        Teuchos::Array<Scalar> values;
        typename Teuchos::Array<Scalar>::iterator value_iterator;
        Teuchos::Array<GO> indices;
        typename Teuchos::Array<GO>::iterator index_iterator;

        for ( int i = 0; i < local_num_rows; ++i )
        {
            // Reset the filter sum.
            filter_sum = 0.0;

            // Get the global row.
            global_row = MT::getGlobalRow( matrix, i );

            // Allocate index and value memory for this row.
            indices.resize( max_entries );
            values.resize( max_entries );

            // Get the values and indices for this row
            MT::getGlobalRowCopy( matrix, 
                                  global_row,
                                  indices(),
                                  values(),
                                  num_entries );

            // Check for degeneracy.
            MCLS_CHECK( num_entries > 0 );

            // Resize local index and value arrays for this row.
            indices.resize( num_entries );
            values.resize( num_entries );

            // If this row contains an entry on the column, add 1 for the
            // identity matrix (H = I-A).
            index_iterator = std::find( indices.begin(), indices.end(),
                                        global_row );
            if ( index_iterator != indices.end() )
            {
                indices[ std::distance(indices.begin(),index_iterator) ] += 1.0;
            }

            // Apply the filter tolerance.
            for ( value_iterator = values.begin(),
                  index_iterator = indices.begin();
                  value_iterator != values.end();
                  ++value_iterator, ++index_iterator )
            {
                if ( std::abs(*value_iterator) < filter_tol )
                {
                    filter_sum += std::abs(*value_iterator);
                    *value_iterator = 0.0;
                    *index_iterator = -1;
                }
            }
            value_iterator = std::remove( values.begin(), values.end(), 0.0 );
            values.resize( std::distance(values.begin(),value_iterator) );
            index_iterator = std::remove( indices.begin(), indices.end(), -1 );
            indices.resize( std::distance(indices.begin(),index_iterator) );

            // Apply the fill level.
            if ( values.size() > fill_value )
            {
                // Get the fill value cutoff.
                sorted_values.resize(values.size() );
                std::copy( values.begin(), values.end(), sorted_values.begin() );
                for( sorted_values_it = sorted_values.begin();
                     sorted_values_it != sorted_values.end();
                     ++sorted_values_it )
                {
                    *sorted_values_it = std::abs( *sorted_values_it );
                }
                std::nth_element( sorted_values.begin(), 
                                  sorted_values.end()-fill_value,
                                  sorted_values.end() );

                // Filter any values below the fill value cutoff.
                for ( value_iterator = values.begin(),
                      index_iterator = indices.begin();
                      value_iterator != values.end();
                      ++value_iterator, ++index_iterator )
                {
                    if ( std::abs(*value_iterator) <
                         *(sorted_values.end()-fill_value) )
                    {
                        filter_sum += std::abs(*value_iterator);
                        *value_iterator = 0.0;
                        *index_iterator = -1;
                    }
                }
                value_iterator = 
                    std::remove( values.begin(), values.end(), 0.0 );
                values.resize( std::distance(values.begin(),value_iterator) );
                index_iterator = 
                    std::remove( indices.begin(), indices.end(), -1 );
                indices.resize( std::distance(indices.begin(),index_iterator) );
            }

            // Check again for degeneracy and consistency.
            MCLS_CHECK( values.size() > 0 );
            MCLS_CHECK( values.size() == indices.size() );

            // Add the values to the reduced iteration matrix.
            reduced_H->insertGlobalValues( global_row, indices(), values() );

            // Add the filter sum to the weight recovery vector.
            rweights_view[i] = filter_sum*weight_recovery;
        }

        reduced_H->fillComplete();
        MCLS_CHECK( reduced_H->isFillComplete() );
    }
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

#endif // end MCLS_TPETRAMATRIXALGORITHMS_HPP

//---------------------------------------------------------------------------//
// end MCLS_TpetraMatrixAlgorithms.hpp
//---------------------------------------------------------------------------//
