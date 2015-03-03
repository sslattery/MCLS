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
 * \file MCLS_MultiSetLinearProblem.hpp
 * \author Stuart R. Slattery
 * \brief Multiple Set Linear Problem declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MULTISETLINEARPROBLEM_HPP
#define MCLS_MULTISETLINEARPROBLEM_HPP

#include "MCLS_config.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_LinearProblem.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_Time.hpp>
#include <Teuchos_Comm.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MultiSetLinearProblem
 * \brief Linear system container for for multiple sets.
 */
template<class Vector, class Matrix>
class MultiSetLinearProblem
{
  public:

    //@{
    //! Typedefs.
    typedef Vector                                      vector_type;
    typedef VectorTraits<Vector>                        VT;
    typedef typename VT::scalar_type                    Scalar;
    typedef Matrix                                      matrix_type;
    typedef MatrixTraits<Vector,Matrix>                 MT;
    //@}

    // Linear problem constructor.
    MultiSetLinearProblem(
	const Teuchos::RCP<const Teuchos::Comm<int> >& global_comm,
	const int num_sets,
	const int set_id,
	const Teuchos::RCP<LinearProblem<Vector,Matrix> >& problem );

    // Operator constructor.
    MultiSetLinearProblem(
	const Teuchos::RCP<const Teuchos::Comm<int> >& global_comm,
	const int num_sets,
	const int set_id,
	const Teuchos::RCP<const Matrix>& A,
	const Teuchos::RCP<Vector>& x,
	const Teuchos::RCP<const Vector>& b );

    // Get the global rank.
    int globalRank() const
    { return d_global_comm->getRank(); }
    
    // Get the number of sets.
    int numSets() const
    { return d_num_sets; }
    
    // Get the set id.
    int setID() const
    { return d_set_id; }

    // Get the block id.
    int blockID() const
    { return d_block_id; }

    // Get the linear problem for the local set.
    Teuchos::RCP<LinearProblem<Vector,Matrix> > getProblem() const
    { return d_problem; }

    // Sum a vector over sets (block-constant). Each set receives the
    // resulting sum. The vector must be in the same parallel decomposition as
    // the LHS and RHS of the set linear problem.
    void blockConstantVectorSum( const Teuchos::RCP<Vector>& vector ) const;

  private:

    // Build the set-constant and block-constant communicators.
    void buildCommunicators();
    
  private:

    // Global communicator.
    Teuchos::RCP<const Teuchos::Comm<int> > d_global_comm;

    // Set-constant communicator.
    Teuchos::RCP<const Teuchos::Comm<int> > d_set_comm;

    // Block-constant communicator.
    Teuchos::RCP<const Teuchos::Comm<int> > d_block_comm;

    // Total number of sets.
    int d_num_sets;
    
    // Local set id.
    int d_set_id;

    // Local block id.
    int d_block_id;

   // Linear problem.
    Teuchos::RCP<LinearProblem<Vector,Matrix> > d_problem;

#if HAVE_MCLS_TIMERS
    // Block-constant vector sum timer.
    Teuchos::RCP<Teuchos::Time> d_bcvs_timer;
#endif
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_MultiSetLinearProblem_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MULTISETLINEARPROBLEM_HPP

//---------------------------------------------------------------------------//
// end MCLS_MultiSetLinearProblem.hpp
//---------------------------------------------------------------------------//

