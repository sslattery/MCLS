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
 * \file MCLS_CommTree.hpp
 * \author Stuart R. Slattery
 * \brief Binary communication tree class definition.
 */
//---------------------------------------------------------------------------//

#ifndef MCLS_COMMTREE_HPP
#define MCLS_COMMTREE_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>

namespace MCLS
{
//---------------------------------------------------------------------------//
/*!
 * \class CommTree
 * \brief Binary communication tree.
 */
//---------------------------------------------------------------------------//
class CommTree
{
  public:

    //@{
    //! Typedefs.
    typedef Teuchos::Comm<int>                        Comm;
    typedef Teuchos::CommRequest<int>                 Request;
    //@}

    //! Constructor.
    CommTree( const Teuchos::RCP<const Comm>& comm );

    // Destructor.
    ~CommTree();

    // Get my id.
    int self() const
    { return d_comm->getRank(); }

    // Get the parent.
    int parent() const
    { return d_parent; }

    //! Get the children.
    std::pair<int,int> children() const
    { return d_children; }

  private:

    // Communicator.
    Teuchos::RCP<const Comm> d_comm;

    // Parent process.
    int d_parent;

    // Child processes.
    std::pair<int,int> d_children;
};

//---------------------------------------------------------------------------//

#endif // end MCLS_COMMTREE_HPP

//---------------------------------------------------------------------------//
// end MCLS_CommTree.hpp
//---------------------------------------------------------------------------//

