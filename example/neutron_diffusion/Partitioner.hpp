//---------------------------------------------------------------------------//
/*!
 * \file Partioner.hpp
 * \author Stuart R. Slattery
 * \brief Mesh partitioner declaration.
 */
//---------------------------------------------------------------------------//

#ifndef MCLSEXAMPLES_PARTITIONER_HPP
#define MCLSEXAMPLES_PARTITIONER_HPP

#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>

namespace MCLSExamples
{

class Partitioner
{
  public:

    //@{
    //! Typedefs.
    typedef unsigned int                          size_type;
    typedef unsigned long                         long_type;
    typedef std::vector<double>                   VecDbl;
    typedef std::pair<size_type,size_type>        SizePair;
    typedef std::pair<long_type,long_type>        LongPair;
    typedef std::pair<double,double>              DblPair;
    typedef std::pair<VecDbl, VecDbl>             VecPair;
    typedef Teuchos::Comm<int>                    CommType;
    typedef Teuchos::RCP<const CommType>          RCP_Comm;
    typedef Teuchos::RCP<Teuchos::ParameterList>  RCP_ParameterList;
    //@}

    // Constructor.
    Partitioner( const RCP_Comm &comm, const RCP_ParameterList &plist );

    // Destructor.
    ~Partitioner();

    //! Get the number of blocks.
    const SizePair& getNumBlocks() const
    { return d_num_blocks; }

    //! Get my block ids.
    const SizePair& getMyBlocks() const
    { return d_my_blocks; }

    //! Get the global edge vectors.
    const VecPair& getGlobalEdges() const
    { return d_global_edges; }

    //! Get the i-indices (vertex-based global ids).
    Teuchos::ArrayView<int> getLocalI()
    { return d_local_i(); }

    //! Get the j-indices (vertex-based global ids).
    Teuchos::ArrayView<int> getLocalJ()
    { return d_local_j(); }

    //! Get the local rows (vertex-based global ids).
    Teuchos::ArrayView<int> getLocalRows()
    { return d_local_rows(); }

    //! Get the ghosted global rows (for output).
    Teuchos::ArrayView<int> getGhostLocalRows()
    { return d_ghost_local_rows(); }

    //! Get the cell sizes.
    std::pair<double,double> getCellSizes()
    { return d_cell_size; }

  private:

    // Number of blocks.
    SizePair d_num_blocks;

    // My block ids.
    SizePair d_my_blocks;

    // Global edge vectors.
    VecPair d_global_edges;

    // Cell size.
    std::pair<double,double> d_cell_size;

    // Local i-indices (vertex-based global ids).
    Teuchos::Array<int> d_local_i;

    // Local j-indices (vertex-based global ids).
    Teuchos::Array<int> d_local_j;

    // Local rows (vertex-based global ids).
    Teuchos::Array<int> d_local_rows;

    // Ghosted local rows (for output).
    Teuchos::Array<int> d_ghost_local_rows;
};

} // end namespace MCLSExamples

#endif // end MCLSEXAMPLES_PARTITIONER_HPP

//---------------------------------------------------------------------------//
// end Partitioner.hpp
//---------------------------------------------------------------------------//

