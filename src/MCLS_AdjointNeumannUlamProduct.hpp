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
 * \file MCLS_AdjointNeumannUlamProduct.hpp
 * \author Stuart R. Slattery
 * \brief Linear Problem declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_ADJOINTNEUMANNULAMPRODUCT_HPP
#define MCLS_ADJOINTNEUMANNULAMPRODUCT_HPP

#include <MCLS_History.hpp>
#include <MCLS_SamplingTools.hpp>

#include <Teuchos_RCP.hpp>
#include <Teuchos_Hashtable.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class AdjointNeumannUlamProduct
 * \brief Adjoint Neumann-Ulam Product of a matrix.
 *
 * H^T = I - A^T 
 * H^T = (P) x (W)
 */
template<class Scalar, class GO>
class AdjointNeumannUlamProduct
{
  public:

    //@{
    //! Typedefs.
    typedef Scalar                                  scalar_type;
    typedef GO                                      global_ordinal_type;
    //@}

    // Constructor.
    template<class Matrix>
    AdjointNeumannUlamProduct( const Teuchos::RCP<const Matrix>& A );

    // Destructor.
    ~AdjointNeumannUlamProduct();

    // Given a history, process a transition to a new state.
    inline void processTransition( History<Scalar,GO>& history );

    // Determine if a given state is on-process.
    inline bool isLocalState( const GO &state );

  private:

    // Local weight table.
    Teuchos::Hashtable<GO,Scalar> d_weight_table;

    // Local CDF table.
    Teuchos::Hashtable<GO,Teuchos::Array<GO> > d_cdf_table;

    // Local column table.
    Teuchos::Hashtable<GO,GO> d_l2g_col;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Inline functions.
//---------------------------------------------------------------------------//
/*!
 * \brief Given an initial state, process a transition to a new state.
 */
template<class Scalar, class GO>
inline void AdjointNeumannUlamProduct<Scalar,GO>::processTransition( 
    History<Scalar,GO>& history )
{

}

//---------------------------------------------------------------------------//
/*!
 * \brief Determine if a given state is on-process.
 */
template<class Scalar, class GO>
inline bool 
AdjointNeumannUlamProduct<Scalar,GO>::isLocalState( const GO& state )
{
    return d_weight_table.containsKey( state );
}

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_AdjointNeumannUlamProduct_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_ADJOINTNEUMANNULAMPRODUCT_HPP

//---------------------------------------------------------------------------//
// end MCLS_AdjointNeumannUlamProduct.hpp
// ---------------------------------------------------------------------------//

