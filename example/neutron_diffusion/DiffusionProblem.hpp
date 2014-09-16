//---------------------------------------------------------------------------//
/*!
 * \file DiffusionProblem.hpp
 * \author Stuart R. Slattery
 * \brief Diffusion problem declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLSEXAMPLES_DIFFUSIONPROBLEM_HPP
#define MCLSEXAMPLES_DIFFUSIONPROBLEM_HPP

#include "Partitioner.hpp"

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>

#include <Tpetra_CrsMatrix.hpp>
#include <Tpetra_Vector.hpp>

namespace MCLSExamples
{

class DiffusionProblem
{
  public:

    //@{
    //! Typedefs.
    typedef Teuchos::RCP<Partitioner>                   RCP_Partitioner;
    typedef Teuchos::RCP<Teuchos::ParameterList>        RCP_ParameterList;
    typedef Teuchos::Comm<int>                          CommType;
    typedef Teuchos::RCP<const CommType>                RCP_Comm;
    //@}

    // Constructor.
    DiffusionProblem( const RCP_Comm& comm, 
		      const RCP_Partitioner& partitioner,
		      const RCP_ParameterList& plist,
		      bool jacobi_precondition = false );

    // Destructor.
    ~DiffusionProblem();

    // Get the linear operator.
    Teuchos::RCP<Tpetra::CrsMatrix<double,int,int> > getOperator() 
    { return d_A; }

    // Get the left-hand side.
    Teuchos::RCP<Tpetra::Vector<double,int> > getLHS()
    { return d_X; }

    // Get the right-hand side.
    Teuchos::RCP<Tpetra::Vector<double,int> > getRHS()
    { return d_B; }

  private:

    // Linear operator.
    Teuchos::RCP<Tpetra::CrsMatrix<double,int,int> > d_A;

    // Left-hand side.
    Teuchos::RCP<Tpetra::Vector<double,int> > d_X;

    // Right-hand side.
    Teuchos::RCP<Tpetra::Vector<double,int> > d_B;
};

} // end namespace MCLSExamples

#endif // end MCLSEXAMPLES_DIFFUSIONPROBLEM_HPP

//---------------------------------------------------------------------------//
// end DiffusionProblem.hpp
//---------------------------------------------------------------------------//

