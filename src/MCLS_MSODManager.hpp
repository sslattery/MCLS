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
 * \file MCLS_SetManager.hpp
 * \author Stuart R. Slattery
 * \brief Multiple set manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_SETMANAGER_HPP
#define MCLS_SETMANAGER_HPP

#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_VectorExport.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class SetManager \brief Class for generating and managing the MSOD
 * decomposition.
 */
template<class Domain>
class SetManager
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                      domain_type;
    typedef Teuchos::Comm<int>                          Comm;
    //@}

    // Constructor.
    SetManager(	const Teuchos::RCP<Domain>& primary_domain,
		const Teuchos::RCP<const Comm>& global_comm,
		Teuchos::ParameterList& plist );

    //! Destructor.
    ~SetManager { /* ... */ }

    //! Get the local domain.
    Teuchos::RCP<Domain> localDomain() const { return d_local_domain; }

    //! Get the number of sets.
    int numSets() const { return d_num_sets; }

    //! Get the size of a set.
    int setSize() const { return d_set_size; }

    //! Set ID for this set for this proc.
    int setID() const { return d_set_id; }

    //! Get the set-constant communication.
    Teuchos::RCP<const Comm> setComm() const { return d_set_comm; }

    //! Get the number of blocks.
    int numBlocks() const { return d_num_blocks; }

    //! Get the size of a block.
    int blockSize() const { return d_block_size; }

    //! Block ID for this block for this proc.
    int blockID() const { return d_block_id; }

    //! Get the block-constant communication.
    Teuchos::RCP<const Comm> blockComm() const { return d_block_comm; }

  private:

    // Build the set-constant communicators.
    void buildSetComms();

    // Build the block-constant communicators.
    void buildBlockComms();

    // Build the global decomposition by broadcasting the primary domain.
    void buildDecomposition();

  private:

    // Global communicator.
    Teuchos::RCP<const Comm> d_global_comm;

    // Number of sets in the problem.
    int d_num_sets;

    // Number of blocks in the problem.
    int d_num_blocks;

    // Size of a set.
    int d_set_size;

    // Size of a block.
    int d_block_size;

    // Set ID for this set.
    int d_set_id;

    // Block ID for this block.
    int d_block_id;

    // Local domain for this proc.
    Teuchos::RCP<Domain> d_local_domain;

    // Set-constant communicator.
    Teuchos::RCP<const Comm> d_set_comm;

    // Block-constant communicator.
    Teuchos::RCP<const Comm> d_block_comm;

    // Primary-to-secondary set exporters.
    Teuchos::Array<VectorExport<Vector> > d_p_to_s_exports;

    // Secondary-to-primary set exporters.
    Teuchos::Array<VectorExport<Vector> > d_s_to_p_exports;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_SetManager_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_SETMANAGER_HPP

//---------------------------------------------------------------------------//
// end MCLS_SetManager.hpp
// ---------------------------------------------------------------------------//

