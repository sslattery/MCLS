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
 * \file MCLS_MSODManager.hpp
 * \author Stuart R. Slattery
 * \brief Multiple-set overlapping-domain decomposition manager declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_MSODMANAGER_HPP
#define MCLS_MSODMANAGER_HPP

#include "MCLS_LinearProblem.hpp"
#include "MCLS_VectorTraits.hpp"
#include "MCLS_MatrixTraits.hpp"
#include "MCLS_VectorExport.hpp"
#include "MCLS_RNGControl.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>

namespace MCLS
{

//---------------------------------------------------------------------------//
/*!
 * \class MSODManager 
 * \brief Class for generating and managing the MSOD decomposition.
 */
template<class Domain, class Source>
class MSODManager
{
  public:

    //@{
    //! Typedefs.
    typedef Domain                                      domain_type;
    typedef Source                                      source_type;
    typedef Teuchos::Comm<int>                          Comm;
    //@}

    // Constructor.
    MSODManager( const Teuchos::RCP<Domain>& primary_domain,
		 const Teuchos::RCP<Source>& primary_source,
		 const Teuchos::RCP<const Comm>& global_comm,
		 const Teuchos::RCP<Teuchos::ParameterList>& plist );

    //! Destructor.
    ~MSODManager { /* ... */ }

    // Update the local domain.
    void updateDomain( const Teuchos::RCP<Domain>& primary_domain );

    // Update the local source.
    void updateSource( const Teuchos::RCP<Source>& primary_source,
		       const Teuchos::RCP<RNGControl>& rng_control );

    //! Get the local domain.
    Teuchos::RCP<Domain> localDomain() const { return d_local_domain; }

    //! Get the local source.
    Teuchos::RCP<Source> localSource() const { return d_local_source; }

    //! Get the number of sets.
    int numSets() const { return d_num_sets; }

    //! Get the number of blocks.
    int numBlocks() const { return d_num_blocks; }

    //! Get the size of a set.
    int setSize() const { return d_set_size; }

    //! Get the size of a block.
    int blockSize() const { return d_block_size; }

    //! Set ID for this set for this proc.
    int setID() const { return d_set_id; }

    //! Block ID for this block for this proc.
    int blockID() const { return d_block_id; }

    //! Get the set-constant communication.
    Teuchos::RCP<const Comm> setComm() const { return d_set_comm; }

    //! Get the block-constant communication.
    Teuchos::RCP<const Comm> blockComm() const { return d_block_comm; }

  private:

    // Build the set-constant communicators.
    void buildSetComms();

    // Build the block-constant communicators.
    void buildBlockComms();

    // Build the global decomposition by broadcasting the primary domain. 
    void broadcastDomain();

    // Build the global decomposition by broadcasting the primary source. 
    void broadcastSource( const Teuchos::RCP<RNGControl>& rng_control );

  private:

    // Global communicator.
    Teuchos::RCP<const Comm> d_global_comm;

    // Parameter list.
    Teuchos::RCP<Teuchos::ParameterList> d_plist;

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

    // Local source for this proc.
    Teuchos::RCP<Source> d_local_source;

    // Set-constant communicator.
    Teuchos::RCP<const Comm> d_set_comm;

    // Block-constant communicator.
    Teuchos::RCP<const Comm> d_block_comm;
};

//---------------------------------------------------------------------------//

} // end namespace MCLS

//---------------------------------------------------------------------------//
// Template includes.
//---------------------------------------------------------------------------//

#include "MCLS_MSODManager_impl.hpp"

//---------------------------------------------------------------------------//

#endif // end MCLS_MSODMANAGER_HPP

//---------------------------------------------------------------------------//
// end MCLS_MSODManager.hpp
// ---------------------------------------------------------------------------//

