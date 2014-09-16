//---------------------------------------------------------------------------//
/*!
 * \file DiffusionProblem.cpp
 * \author Stuart R. Slattery
 * \brief Diffusion problem implementation.
 */
//---------------------------------------------------------------------------//

#include <iostream>
#include <cmath>

#include "DiffusionProblem.hpp"

#include <MCLS_DBC.hpp>

#include <Teuchos_ArrayView.hpp>

#include <Tpetra_Map.hpp>

namespace MCLSExamples
{

//---------------------------------------------------------------------------//
/*!
 * \brief Constructor.
 */
DiffusionProblem::DiffusionProblem( const RCP_Comm& comm, 
				    const RCP_Partitioner& partitioner,
				    const RCP_ParameterList& plist,
				    bool jacobi_precondition )
{
    // Get the local rows from the partitioner.
    Teuchos::ArrayView<int> local_rows = partitioner->getLocalRows();
    Teuchos::ArrayView<int> local_i = partitioner->getLocalI();
    Teuchos::ArrayView<int> local_j = partitioner->getLocalJ();
    Teuchos::ArrayView<int>::const_iterator local_i_it;
    Teuchos::ArrayView<int>::const_iterator local_j_it;

    // Build the map.
    Teuchos::RCP<const Tpetra::Map<int> > row_map = 
	    Tpetra::createNonContigMap<int,int>( local_rows, comm );

    // Build the operator.
    d_A = Tpetra::createCrsMatrix<double,int,int>( row_map, 9 );

    int N = partitioner->getGlobalEdges().first.size();
    double dx = partitioner->getCellSizes().first;
    double dy = partitioner->getCellSizes().second;
    double xs_s = plist->get<double>("SCATTERING XS");
    double xs_a = plist->get<double>("ABSORPTION XS");
    double D = 1.0 / ( 3.0*(xs_a+xs_s) );
    double d_length = std::sqrt( 1.0 / (3*xs_a*(xs_a+xs_s)) );

    if ( comm->getRank() == 0 )
    {
	std::cout << std::endl;
        std::cout << "Diffusion Length " << d_length << std::endl;
        std::cout << "Mesh Size " << dx << std::endl;
        std::cout << "L/dx " << d_length/dx << std::endl;
    }

    double diag = xs_a + D*10.0/(3.0*dx*dx);
    double iminus1 = -2.0*D/(3.0*dx*dx);
    double iplus1 = -2.0*D/(3.0*dx*dx);
    double jminus1 = -2.0*D/(3.0*dy*dy);
    double jplus1 = -2.0*D/(3.0*dy*dy);
    double iminus1jminus1 = -D/(6.0*dx*dx);
    double iplus1jminus1 = -D/(6.0*dx*dx);
    double iminus1jplus1 = -D/(6.0*dy*dy);
    double iplus1jplus1 = -D/(6.0*dy*dy);

    // Apply preconditioning.
    double jacobi_scale = diag;
    if ( jacobi_precondition )
    {
	iminus1 /= jacobi_scale;
	iplus1 /= jacobi_scale;
	jminus1 /= jacobi_scale;
	jplus1 /= jacobi_scale;
	iminus1jminus1 /= jacobi_scale;
	iplus1jminus1 /= jacobi_scale;
	iminus1jplus1 /= jacobi_scale;
	iplus1jplus1 /= jacobi_scale;
	diag /= jacobi_scale;
    }

    int idx;
    int idx_iminus1;
    int idx_iplus1;
    int idx_jminus1;
    int idx_jplus1;
    int idx_iminus1jminus1;
    int idx_iplus1jminus1;
    int idx_iminus1jplus1;
    int idx_iplus1jplus1;

    // Determine if the local mesh is on a boundary.
    bool has_lo_x = false;
    bool has_lo_y = false;
    bool has_hi_x = false;
    bool has_hi_y = false;
    if ( partitioner->getMyBlocks().first == 0 )
    {
        has_lo_x = true;
    }
    if ( partitioner->getMyBlocks().second == 0 )
    {
        has_lo_y = true;
    }
    if ( partitioner->getMyBlocks().first == partitioner->getNumBlocks().first-1 )
    {
        has_hi_x = true;
    }
    if ( partitioner->getMyBlocks().second == partitioner->getNumBlocks().second-1 )
    {
        has_hi_y = true;
    }

    // Vacuum boundaries.
    Teuchos::Array<int> corner_idx(4);
    Teuchos::Array<double> corner_values(4);

    // Lower left boundary vacuum (nonreentrant current).
    if ( has_lo_x && has_lo_y )
    {
        int i = 0;
        int j = 0;

        idx                = i + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        corner_idx[0] = idx;
        corner_idx[1] = idx_iplus1;
        corner_idx[2] = idx_jplus1;
        corner_idx[3] = idx_iplus1jplus1;

        corner_values[0] = diag;
        corner_values[1] = iplus1;
        corner_values[2] = jplus1;
        corner_values[3] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, corner_idx(), corner_values() );
    }

    // Lower right boundary vacuum (nonreentrant current).
    if ( has_hi_x && has_lo_y )
    {
        int i = N-1;
        int j = 0;

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;

        corner_idx[0] = idx;
        corner_idx[1] = idx_iminus1;
        corner_idx[2] = idx_jplus1;
        corner_idx[3] = idx_iminus1jplus1;

        corner_values[0] = diag;
        corner_values[1] = iminus1;
        corner_values[2] = jplus1;
        corner_values[3] = iminus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, corner_idx(), corner_values() );
    }

    // Upper left boundary vacuum (nonreentrant current).
    if ( has_lo_x && has_hi_y )
    {
        int i = 0;
        int j = N-1;

        idx                = i + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;

        corner_idx[0] = idx;
        corner_idx[1] = idx_iplus1;
        corner_idx[2] = idx_jminus1;
        corner_idx[3] = idx_iplus1jminus1;

        corner_values[0] = diag;
        corner_values[1] = iplus1;
        corner_values[2] = jminus1;
        corner_values[3] = iplus1jminus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, corner_idx(), corner_values() );
    }

    // Upper right boundary vacuum (nonreentrant current).
    if ( has_hi_x && has_hi_y )
    {
        int i = N-1;
        int j = N-1;

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;

        corner_idx[0] = idx;
        corner_idx[1] = idx_iminus1;
        corner_idx[2] = idx_jminus1;
        corner_idx[3] = idx_iminus1jminus1;

        corner_values[0] = diag;
        corner_values[1] = iminus1;
        corner_values[2] = jminus1;
        corner_values[3] = iminus1jminus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, corner_idx(), corner_values() );
    }

    Teuchos::Array<int> bnd_idx(6);
    Teuchos::Array<double> bnd_values(6);

    // Min X boundary vacuum (nonreentrant current).
    if ( has_lo_x )
    {
        for ( local_j_it = local_j.begin()+1; 
              local_j_it != local_j.end()-1;
              ++local_j_it )
        {
            int i = 0;
            int j = *local_j_it;

            idx                = i + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            bnd_idx[0] = idx;
            bnd_idx[1] = idx_iplus1;
            bnd_idx[2] = idx_jminus1;
            bnd_idx[3] = idx_jplus1;
            bnd_idx[4] = idx_iplus1jminus1;
            bnd_idx[5] = idx_iplus1jplus1;

            bnd_values[0] = diag;
            bnd_values[1] = iplus1;
            bnd_values[2] = jminus1;
            bnd_values[3] = jplus1;
            bnd_values[4] = iplus1jminus1;
            bnd_values[5] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
        }
    }
    if ( has_lo_x && !has_lo_y )
    {
        int i = 0;
        int j = local_j.front();

        idx                = i + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iplus1;
        bnd_idx[2] = idx_jminus1;
        bnd_idx[3] = idx_jplus1;
        bnd_idx[4] = idx_iplus1jminus1;
        bnd_idx[5] = idx_iplus1jplus1;

        bnd_values[0] = diag;
        bnd_values[1] = iplus1;
        bnd_values[2] = jminus1;
        bnd_values[3] = jplus1;
        bnd_values[4] = iplus1jminus1;
        bnd_values[5] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }
    if ( has_lo_x && !has_hi_y )
    {
        int i = 0;
        int j = local_j.back();

        idx                = i + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iplus1;
        bnd_idx[2] = idx_jminus1;
        bnd_idx[3] = idx_jplus1;
        bnd_idx[4] = idx_iplus1jminus1;
        bnd_idx[5] = idx_iplus1jplus1;

        bnd_values[0] = diag;
        bnd_values[1] = iplus1;
        bnd_values[2] = jminus1;
        bnd_values[3] = jplus1;
        bnd_values[4] = iplus1jminus1;
        bnd_values[5] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }

    // Max X boundary vacuum (nonreentrant current).
    if ( has_hi_x )
    {
        for ( local_j_it = local_j.begin()+1;
              local_j_it != local_j.end()-1;
              ++local_j_it )
        {
            int i = N-1;
            int j = *local_j_it;

            idx = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;

            bnd_idx[0] = idx;
            bnd_idx[1] = idx_iminus1;
            bnd_idx[2] = idx_jminus1;
            bnd_idx[3] = idx_jplus1;
            bnd_idx[4] = idx_iminus1jminus1;
            bnd_idx[5] = idx_iminus1jplus1;

            bnd_values[0] = diag;
            bnd_values[1] = iminus1;
            bnd_values[2] = jminus1;
            bnd_values[3] = jplus1;
            bnd_values[4] = iminus1jminus1;
            bnd_values[5] = iminus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
        }
    }
    if ( has_hi_x && !has_lo_y )
    {
        int i = N-1;
        int j = local_j.front();

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iminus1;
        bnd_idx[2] = idx_jminus1;
        bnd_idx[3] = idx_jplus1;
        bnd_idx[4] = idx_iminus1jminus1;
        bnd_idx[5] = idx_iminus1jplus1;

        bnd_values[0] = diag;
        bnd_values[1] = iminus1;
        bnd_values[2] = jminus1;
        bnd_values[3] = jplus1;
        bnd_values[4] = iminus1jminus1;
        bnd_values[5] = iminus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }
    if ( has_hi_x && !has_hi_y )
    {
        int i = N-1;
        int j = local_j.back();

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iminus1;
        bnd_idx[2] = idx_jminus1;
        bnd_idx[3] = idx_jplus1;
        bnd_idx[4] = idx_iminus1jminus1;
        bnd_idx[5] = idx_iminus1jplus1;

        bnd_values[0] = diag;
        bnd_values[1] = iminus1;
        bnd_values[2] = jminus1;
        bnd_values[3] = jplus1;
        bnd_values[4] = iminus1jminus1;
        bnd_values[5] = iminus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }

    // Min Y boundary vacuum (nonreentrant current).
    if ( has_lo_y )
    {
        for ( local_i_it = local_i.begin()+1;
              local_i_it != local_i.end()-1;
              ++local_i_it )
        {
            int i = *local_i_it;
            int j = 0;

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            bnd_idx[0] = idx;
            bnd_idx[1] = idx_iminus1;
            bnd_idx[2] = idx_iplus1;
            bnd_idx[3] = idx_jplus1;
            bnd_idx[4] = idx_iminus1jplus1;
            bnd_idx[5] = idx_iplus1jplus1;

            bnd_values[0] = diag;
            bnd_values[1] = iminus1;
            bnd_values[2] = iplus1;
            bnd_values[3] = jplus1;
            bnd_values[4] = iminus1jplus1;
            bnd_values[5] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
        }
    }
    if ( has_lo_y && !has_lo_x )
    {
        int i = local_i.front();
        int j = 0;

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iminus1;
        bnd_idx[2] = idx_iplus1;
        bnd_idx[3] = idx_jplus1;
        bnd_idx[4] = idx_iminus1jplus1;
        bnd_idx[5] = idx_iplus1jplus1;

        bnd_values[0] = diag;
        bnd_values[1] = iminus1;
        bnd_values[2] = iplus1;
        bnd_values[3] = jplus1;
        bnd_values[4] = iminus1jplus1;
        bnd_values[5] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }
    if ( has_lo_y && !has_hi_x )
    {
        int i = local_i.back();
        int j = 0;

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iminus1;
        bnd_idx[2] = idx_iplus1;
        bnd_idx[3] = idx_jplus1;
        bnd_idx[4] = idx_iminus1jplus1;
        bnd_idx[5] = idx_iplus1jplus1;

        bnd_values[0] = diag;
        bnd_values[1] = iminus1;
        bnd_values[2] = iplus1;
        bnd_values[3] = jplus1;
        bnd_values[4] = iminus1jplus1;
        bnd_values[5] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }

    // Max Y boundary vacuum (nonreentrant current).
    if ( has_hi_y )
    {
        for ( local_i_it = local_i.begin()+1;
              local_i_it != local_i.end()-1;
              ++local_i_it )
        {
            int i = *local_i_it;
            int j = N-1;

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;

            bnd_idx[0] = idx;
            bnd_idx[1] = idx_iminus1;
            bnd_idx[2] = idx_iplus1;
            bnd_idx[3] = idx_jminus1;
            bnd_idx[4] = idx_iminus1jminus1;
            bnd_idx[5] = idx_iplus1jminus1;

            bnd_values[0] = diag;
            bnd_values[1] = iminus1;
            bnd_values[2] = iplus1;
            bnd_values[3] = jminus1;
            bnd_values[4] = iminus1jminus1;
            bnd_values[5] = iplus1jminus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
        }
    }
    if ( has_hi_y && !has_lo_x )
    {
        int i = local_i.front();
        int j = N-1;

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iminus1;
        bnd_idx[2] = idx_iplus1;
        bnd_idx[3] = idx_jminus1;
        bnd_idx[4] = idx_iminus1jminus1;
        bnd_idx[5] = idx_iplus1jminus1;

        bnd_values[0] = diag;
        bnd_values[1] = iminus1;
        bnd_values[2] = iplus1;
        bnd_values[3] = jminus1;
        bnd_values[4] = iminus1jminus1;
        bnd_values[5] = iplus1jminus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }
    if ( has_hi_y && !has_hi_x )
    {
        int i = local_i.back();
        int j = N-1;

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;

        bnd_idx[0] = idx;
        bnd_idx[1] = idx_iminus1;
        bnd_idx[2] = idx_iplus1;
        bnd_idx[3] = idx_jminus1;
        bnd_idx[4] = idx_iminus1jminus1;
        bnd_idx[5] = idx_iplus1jminus1;

        bnd_values[0] = diag;
        bnd_values[1] = iminus1;
        bnd_values[2] = iplus1;
        bnd_values[3] = jminus1;
        bnd_values[4] = iminus1jminus1;
        bnd_values[5] = iplus1jminus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, bnd_idx(), bnd_values() );
    }

    // Central grid points
    Teuchos::Array<int> center_idx(9);
    Teuchos::Array<double> center_values(9);
    for ( local_j_it = local_j.begin()+1;
          local_j_it != local_j.end()-1;
          ++local_j_it )
    {
        for ( local_i_it = local_i.begin()+1;
              local_i_it != local_i.end()-1;
              ++local_i_it )
        {
            int i = *local_i_it;
            int j = *local_j_it;

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            center_idx[0] = idx;
            center_idx[1] = idx_iminus1;
            center_idx[2] = idx_iplus1;
            center_idx[3] = idx_jminus1;
            center_idx[4] = idx_jplus1;
            center_idx[5] = idx_iminus1jminus1;
            center_idx[6] = idx_iplus1jminus1;
            center_idx[7] = idx_iminus1jplus1;
            center_idx[8] = idx_iplus1jplus1;

            center_values[0] = diag;
            center_values[1] = iminus1;
            center_values[2] = iplus1;
            center_values[3] = jminus1;
            center_values[4] = jplus1;
            center_values[5] = iminus1jminus1;
            center_values[6] = iplus1jminus1;
            center_values[7] = iminus1jplus1;
            center_values[8] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, center_idx(), center_values() );
        }
    }
    if ( !has_lo_x )
    {
        for ( local_j_it = local_j.begin()+1;
              local_j_it != local_j.end()-1;
              ++local_j_it )
        {
            int i = local_i.front();
            int j = *local_j_it;

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            center_idx[0] = idx;
            center_idx[1] = idx_iminus1;
            center_idx[2] = idx_iplus1;
            center_idx[3] = idx_jminus1;
            center_idx[4] = idx_jplus1;
            center_idx[5] = idx_iminus1jminus1;
            center_idx[6] = idx_iplus1jminus1;
            center_idx[7] = idx_iminus1jplus1;
            center_idx[8] = idx_iplus1jplus1;

            center_values[0] = diag;
            center_values[1] = iminus1;
            center_values[2] = iplus1;
            center_values[3] = jminus1;
            center_values[4] = jplus1;
            center_values[5] = iminus1jminus1;
            center_values[6] = iplus1jminus1;
            center_values[7] = iminus1jplus1;
            center_values[8] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, center_idx(), center_values() );
        }
    }
    if ( !has_hi_x )
    {
        for ( local_j_it = local_j.begin()+1;
              local_j_it != local_j.end()-1;
              ++local_j_it )
        {
            int i = local_i.back();
            int j = *local_j_it;

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            center_idx[0] = idx;
            center_idx[1] = idx_iminus1;
            center_idx[2] = idx_iplus1;
            center_idx[3] = idx_jminus1;
            center_idx[4] = idx_jplus1;
            center_idx[5] = idx_iminus1jminus1;
            center_idx[6] = idx_iplus1jminus1;
            center_idx[7] = idx_iminus1jplus1;
            center_idx[8] = idx_iplus1jplus1;

            center_values[0] = diag;
            center_values[1] = iminus1;
            center_values[2] = iplus1;
            center_values[3] = jminus1;
            center_values[4] = jplus1;
            center_values[5] = iminus1jminus1;
            center_values[6] = iplus1jminus1;
            center_values[7] = iminus1jplus1;
            center_values[8] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, center_idx(), center_values() );
        }
    }
    if ( !has_lo_y )
    {
        for ( local_i_it = local_i.begin()+1;
              local_i_it != local_i.end()-1;
              ++local_i_it )
        {
            int i = *local_i_it;
            int j = local_j.front();

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            center_idx[0] = idx;
            center_idx[1] = idx_iminus1;
            center_idx[2] = idx_iplus1;
            center_idx[3] = idx_jminus1;
            center_idx[4] = idx_jplus1;
            center_idx[5] = idx_iminus1jminus1;
            center_idx[6] = idx_iplus1jminus1;
            center_idx[7] = idx_iminus1jplus1;
            center_idx[8] = idx_iplus1jplus1;

            center_values[0] = diag;
            center_values[1] = iminus1;
            center_values[2] = iplus1;
            center_values[3] = jminus1;
            center_values[4] = jplus1;
            center_values[5] = iminus1jminus1;
            center_values[6] = iplus1jminus1;
            center_values[7] = iminus1jplus1;
            center_values[8] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, center_idx(), center_values() );
        }
    }
    if ( !has_hi_y )
    {
        for ( local_i_it = local_i.begin()+1;
              local_i_it != local_i.end()-1;
              ++local_i_it )
        {
            int i = *local_i_it;
            int j = local_j.back();

            idx                = i + j*N;
            idx_iminus1        = (i-1) + j*N;
            idx_iplus1         = (i+1) + j*N;
            idx_jminus1        = i + (j-1)*N;
            idx_jplus1         = i + (j+1)*N;
            idx_iminus1jminus1 = (i-1) + (j-1)*N;
            idx_iplus1jminus1  = (i+1) + (j-1)*N;
            idx_iminus1jplus1  = (i-1) + (j+1)*N;
            idx_iplus1jplus1   = (i+1) + (j+1)*N;

            center_idx[0] = idx;
            center_idx[1] = idx_iminus1;
            center_idx[2] = idx_iplus1;
            center_idx[3] = idx_jminus1;
            center_idx[4] = idx_jplus1;
            center_idx[5] = idx_iminus1jminus1;
            center_idx[6] = idx_iplus1jminus1;
            center_idx[7] = idx_iminus1jplus1;
            center_idx[8] = idx_iplus1jplus1;

            center_values[0] = diag;
            center_values[1] = iminus1;
            center_values[2] = iplus1;
            center_values[3] = jminus1;
            center_values[4] = jplus1;
            center_values[5] = iminus1jminus1;
            center_values[6] = iplus1jminus1;
            center_values[7] = iminus1jplus1;
            center_values[8] = iplus1jplus1;

            MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
            d_A->insertGlobalValues( idx, center_idx(), center_values() );
        }
    }
    if ( !has_lo_x && !has_lo_y )
    {
        int i = local_i.front();
        int j = local_j.front();

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        center_idx[0] = idx;
        center_idx[1] = idx_iminus1;
        center_idx[2] = idx_iplus1;
        center_idx[3] = idx_jminus1;
        center_idx[4] = idx_jplus1;
        center_idx[5] = idx_iminus1jminus1;
        center_idx[6] = idx_iplus1jminus1;
        center_idx[7] = idx_iminus1jplus1;
        center_idx[8] = idx_iplus1jplus1;

        center_values[0] = diag;
        center_values[1] = iminus1;
        center_values[2] = iplus1;
        center_values[3] = jminus1;
        center_values[4] = jplus1;
        center_values[5] = iminus1jminus1;
        center_values[6] = iplus1jminus1;
        center_values[7] = iminus1jplus1;
        center_values[8] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, center_idx(), center_values() );
    }
    if ( !has_lo_x && !has_hi_y )
    {
        int i = local_i.front();
        int j = local_j.back();

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        center_idx[0] = idx;
        center_idx[1] = idx_iminus1;
        center_idx[2] = idx_iplus1;
        center_idx[3] = idx_jminus1;
        center_idx[4] = idx_jplus1;
        center_idx[5] = idx_iminus1jminus1;
        center_idx[6] = idx_iplus1jminus1;
        center_idx[7] = idx_iminus1jplus1;
        center_idx[8] = idx_iplus1jplus1;

        center_values[0] = diag;
        center_values[1] = iminus1;
        center_values[2] = iplus1;
        center_values[3] = jminus1;
        center_values[4] = jplus1;
        center_values[5] = iminus1jminus1;
        center_values[6] = iplus1jminus1;
        center_values[7] = iminus1jplus1;
        center_values[8] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, center_idx(), center_values() );
    }
    if ( !has_hi_x && !has_lo_y )
    {
        int i = local_i.back();
        int j = local_j.front();

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        center_idx[0] = idx;
        center_idx[1] = idx_iminus1;
        center_idx[2] = idx_iplus1;
        center_idx[3] = idx_jminus1;
        center_idx[4] = idx_jplus1;
        center_idx[5] = idx_iminus1jminus1;
        center_idx[6] = idx_iplus1jminus1;
        center_idx[7] = idx_iminus1jplus1;
        center_idx[8] = idx_iplus1jplus1;

        center_values[0] = diag;
        center_values[1] = iminus1;
        center_values[2] = iplus1;
        center_values[3] = jminus1;
        center_values[4] = jplus1;
        center_values[5] = iminus1jminus1;
        center_values[6] = iplus1jminus1;
        center_values[7] = iminus1jplus1;
        center_values[8] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, center_idx(), center_values() );
    }
    if ( !has_hi_x && !has_hi_y )
    {
        int i = local_i.back();
        int j = local_j.back();

        idx                = i + j*N;
        idx_iminus1        = (i-1) + j*N;
        idx_iplus1         = (i+1) + j*N;
        idx_jminus1        = i + (j-1)*N;
        idx_jplus1         = i + (j+1)*N;
        idx_iminus1jminus1 = (i-1) + (j-1)*N;
        idx_iplus1jminus1  = (i+1) + (j-1)*N;
        idx_iminus1jplus1  = (i-1) + (j+1)*N;
        idx_iplus1jplus1   = (i+1) + (j+1)*N;

        center_idx[0] = idx;
        center_idx[1] = idx_iminus1;
        center_idx[2] = idx_iplus1;
        center_idx[3] = idx_jminus1;
        center_idx[4] = idx_jplus1;
        center_idx[5] = idx_iminus1jminus1;
        center_idx[6] = idx_iplus1jminus1;
        center_idx[7] = idx_iminus1jplus1;
        center_idx[8] = idx_iplus1jplus1;

        center_values[0] = diag;
        center_values[1] = iminus1;
        center_values[2] = iplus1;
        center_values[3] = jminus1;
        center_values[4] = jplus1;
        center_values[5] = iminus1jminus1;
        center_values[6] = iplus1jminus1;
        center_values[7] = iminus1jplus1;
        center_values[8] = iplus1jplus1;

        MCLS_CHECK( row_map->isNodeGlobalElement(idx) );
        d_A->insertGlobalValues( idx, center_idx(), center_values() );
    }
    comm->barrier();

    d_A->fillComplete();

    // Build the solution vector.
    double X_val = 0.0;
    d_X = Tpetra::createVector<double,int>( row_map );
    d_X->putScalar( X_val );

    // Build the source.
    double source_strength = plist->get<double>("SOURCE STRENGTH");
    d_B = Tpetra::createVector<double,int>( row_map );

    if ( plist->get<std::string>("SOURCE TYPE") == "UNIFORM" )
    {
	d_B->putScalar( source_strength );
    }
    else if ( plist->get<std::string>("SOURCE TYPE") == "POINT" )
    {
	int source_location = plist->get<int>("SOURCE LOCATION");
	if ( row_map->isNodeGlobalElement( source_location ) )
	{
	    d_B->replaceGlobalValue( source_location, source_strength );
	}
    }

    // Jacobi precondition if necessary.
    if ( jacobi_precondition )
    {
	d_B->scale( 1.0/jacobi_scale );
    }
}

//---------------------------------------------------------------------------//
/*!
 * \brief Destructor.
 */
DiffusionProblem::~DiffusionProblem()
{ /* ... */ }

//---------------------------------------------------------------------------//

} // end namespace MCLSExamples

//---------------------------------------------------------------------------//
// end DiffusionProblem.cpp
//---------------------------------------------------------------------------//

