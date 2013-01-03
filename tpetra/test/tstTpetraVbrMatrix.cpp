//---------------------------------------------------------------------------//
/*!
 * \file tstTpetraVbrMatrix.cpp
 * \author Stuart R. Slattery
 * \brief Tpetra vector tests.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <algorithm>
#include <string>
#include <cassert>

#include <MCLS_VectorTraits.hpp>
#include <MCLS_TpetraAdapter.hpp>

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_ArrayRCP.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ArrayView.hpp>
#include <Teuchos_TypeTraits.hpp>

#include <Tpetra_BlockMap.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_VbrMatrix.hpp>

TEUCHOS_UNIT_TEST( VbrMatrix, test )
{
    typedef Tpetra::VbrMatrix<double,int,int> OperatorType;
    typedef Tpetra::Vector<double,int,int> VectorType;
    typedef MCLS::VectorTraits<double,int,int,VectorType> VT;

    Teuchos::RCP<const Teuchos::Comm<int> > comm = 
	Teuchos::DefaultComm<int>::getComm();
    int comm_size = comm->getSize();

    int local_num_blocks = 10;
    int global_num_blocks = local_num_blocks*comm_size;
    int block_size = 3;
    int index_base = 0;

    Teuchos::RCP<const Tpetra::BlockMap<int,int> > map = Teuchos::rcp(
	new Tpetra::BlockMap<int,int>( global_num_blocks, block_size, index_base, comm ) );

    std::cout << "Global num blocks: " << map->getGlobalNumBlocks() << std::endl;
    std::cout << "Node Block IDs: " << map->getNodeBlockIDs() << std::endl;
    std::cout << "Node First Point in Blocks: " << map->getNodeFirstPointInBlocks()() << std::endl;    
    std::cout << "First global in last: " << map->getFirstGlobalPointInLocalBlock( 9 ) << std::endl;
    std::cout << "Point entries: " << map->getPointMap()->getNodeElementList() << std::endl;

    Teuchos::RCP<Tpetra::VbrMatrix<double,int,int> > A = Teuchos::rcp(
	new Tpetra::VbrMatrix<double,int,int>( map, 0 ) );


}

//---------------------------------------------------------------------------//
// end tstTpetraVbrMatrix.cpp
//---------------------------------------------------------------------------//

